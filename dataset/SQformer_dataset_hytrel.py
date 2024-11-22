import sys

from collections import defaultdict
from dataclasses import dataclass
import logging
from math import remainder
import pathlib
import pickle
import random
from attr import field
from networkx import graph_atlas

import numpy as np
import torch
import transformers
from transformers.optimization import AdamW, get_scheduler

from datasets import concatenate_datasets, load_dataset

import os
from typing import Union, Dict, List, Sequence, Optional
from torch.utils.data import Dataset
from torch_geometric.data.batch import Batch

import torch.nn.functional as F
import datasets
from dataset.data import BipartiteData #StructDataConverter, _get_dist_mat
from utils.utils import load_json

from transformers.data import DataCollatorWithPadding

import zipfile

logger = logging.getLogger(__name__)


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"

DEFAULT_GRAPH_PAD_TOKEN = "[g_patch]"
DEFAULT_CVT_TOKEN = "[CVT]"


def merge_graph(graphs: List[Dict]):
    key_to_lists = defaultdict(list)

    keys = graphs[0].keys()

    for graph in graphs:
        for key in keys:
            lists = key_to_lists[key]
            lists.append(graph[key])

    seq_length = max(node_types.shape[0] for node_types in key_to_lists["node_types"])

    key_to_array = {}
    for key, lists in key_to_lists.items():
        if key == "node_token_ids":
            key_to_array[key] = lists
        elif key in ["node_types", "graph_attention_mask"]:
            key_to_array[key] = torch.nn.utils.rnn.pad_sequence(lists, batch_first=True)
        elif key == "dist_mat":
            dist_mat = [
                F.pad(d, (0, seq_length - d.shape[0], 0, seq_length - d.shape[1]), value=0).long()
                for d in lists
            ]
            key_to_array[key] = torch.stack(dist_mat)

    return key_to_array


def move_tensor_to_device(input_dict, device):
    for key, value in input_dict.items():
        if isinstance(value, dict):
            move_tensor_to_device(value, device)
        elif isinstance(value, torch.Tensor):
            input_dict[key] = value.to(device)


def build_instruction_dataset(
    data_path: Union[List[str], str],
    llm_tokenizer: transformers.PreTrainedTokenizer,
    encoder_tokenizer: transformers.PreTrainedTokenizer,
    max_seq_length: int = 2560,
    max_desc_length: int = 2048,
    max_qformer_length= 64,
    data_cache_dir=None,
    preprocessing_num_workers=64,
    num_query_tokens=10,
    training=True,
    qformer_pretraining=False,
    shuffle_desc=True,
    reprocess=False,
):
    """_summary_

    Args:
        data_path (Union[List[str], str]): _description_
        llm_tokenizer (transformers.PreTrainedTokenizer): tokenizer for down-stream LLM
        encoder_tokenizer (transformers.PreTrainedTokenizer): tokenizer for g-former
        max_seq_length (int, optional): _description_. Defaults to 256.
        max_desc_length: Defaults to 0
        data_cache_dir (_type_, optional): _description_. Defaults to None.
        preprocessing_num_workers (int, optional): _description_. Defaults to 10.
    """

    return GraphDataset(data_path, llm_tokenizer, encoder_tokenizer, num_query_tokens, max_seq_length, max_qformer_length, training, qformer_pretraining)

    assert max_seq_length > max_desc_length

    converter = StructDataConverter(encoder_tokenizer)

    def tokenization(examples):
        targets = []
        questions = []
        inputs = []
        struct_ins = []
        graphs = []

        for label, question, input in zip(
            examples["label"], examples["question"], examples["input"]
        ):
            if num_query_tokens > 0:
                idx = input.rfind('\n\n\n')
                input = input[:idx] + \
                    f'\n\nstruct data representation tokens: {DEFAULT_GRAPH_PAD_TOKEN * num_query_tokens}' + \
                    input[idx:]

            target = f"{label}{llm_tokenizer.eos_token}"

            # question = f'{question} query tokens: {DEFAULT_GRAPH_PAD_TOKEN * num_query_tokens}'
            question = f'{question} query tokens: '

            inputs.append(input)
            targets.append(target)
            questions.append(question)

        graphs = examples['graph']

        # add_special_tokens=False: not add <s>
        tokenized_inputs = llm_tokenizer(
            inputs, return_attention_mask=False, add_special_tokens=False
        )
        tokenized_targets = llm_tokenizer(
            targets, return_attention_mask=False, add_special_tokens=False
        )
        tokenized_questions = encoder_tokenizer(questions)

        all_input_ids = []
        all_labels = []
        all_question_ids = []

        # lens = []

        for input, t, q in zip(
            tokenized_inputs["input_ids"],
            tokenized_targets["input_ids"],
            tokenized_questions["input_ids"],
        ):
            s = [llm_tokenizer.bos_token_id] + input
            if training:
                input_ids = torch.LongTensor(s + t)[:max_seq_length]
                labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            else:
                input_ids = torch.LongTensor(s)[:max_seq_length]
                labels = torch.LongTensor([IGNORE_INDEX] * len(s))[:max_seq_length]

            question_ids = torch.LongTensor(q)
            assert len(input_ids) < max_seq_length
            assert len(input_ids) == len(labels)

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_question_ids.append(question_ids)

            # lens.append(len(s + t))

        # hist, bins = np.histogram(lens, bins=10)
        # for i in range(len(hist)):
        #     print(f"{int(bins[i])} -  {int(bins[i+1])}: {hist[i]}")

        for i, graph in enumerate(graphs):
            num_nodes = len(graph["node_types"])

            graph["node_types"] = torch.LongTensor(graph["node_types"])
            graph["graph_attention_mask"] = torch.LongTensor([1] * num_nodes)
            # -1 -> 0
            graph["node_token_ids"] = torch.LongTensor(graph["node_token_ids"])

        results = {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "question_ids": all_question_ids,
            "graph": graphs,
        }

        return results

    print("building dataset...")
    all_datasets = []

    if not isinstance(data_path, (list, tuple)):
        # convert to str if data_path is Pathlib.Path
        data_path = [str(data_path)]
    for file in data_path:
        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))
        cache_path = os.path.join(
            data_cache_dir,
            os.path.basename(file).split(".")[0] +
            f"{max_desc_length}_{max_seq_length}_{num_query_tokens}_{llm_tokenizer.name_or_path.split('/')[-1]}",
        )
        os.makedirs(cache_path, exist_ok=True)
        try:
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info(f"training datasets-{file} has been loaded from disk")
        except Exception as e:
            raw_dataset = load_dataset("parquet", data_files=file)
            tokenized_dataset = raw_dataset.map(
                tokenization,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=[],
                keep_in_memory=False,
                desc="preprocessing on dataset",
            )
            processed_dataset = tokenized_dataset
            processed_dataset.save_to_disk(cache_path)
        processed_dataset.set_format("torch")
        all_datasets.append(processed_dataset["train"])
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets


class GraphDataset(Dataset):
    def __init__(self, data_path, llm_tokenizer, encoder_tokenizer, num_query_tokens=10, max_seq_length=2560, max_qformer_length=64, training=True, qformer_pretraining=True) -> None:
        self.raw_dataset = load_dataset("parquet", data_files=str(data_path), cache_dir='./data/.cache')['train']
        self.llm_tokenizer = llm_tokenizer
        self.encoder_tokenizer = encoder_tokenizer
        self.num_query_tokens = num_query_tokens
        self.max_seq_length = max_seq_length
        self.max_qformer_length = max_qformer_length
        self.training = training

        self.sqformer_pretraining = qformer_pretraining

    def __getitem__(self, index):
        sample = self.raw_dataset[index]

        if self.sqformer_pretraining:
            assert type(sample['question']) is list
            idx = random.choice(range(len(sample['question'])))

            question = sample['question'][idx]
            label = sample['label'][idx]

            graph = sample['graph']
        else:
            # not used
            question = ''

            if type(sample['input']) is list:
                # pretraining gnn and g-former based on llm
                idx = random.choice(range(len(sample['input'])))

                input_ = sample['input'][idx]
                label = sample['label'][idx]
            else:
                input_ = sample['input']
                label = sample['label']

        if 'graph_path' in sample:
            # use cache
            try:
                with zipfile.ZipFile(sample['graph_path'][0], 'r') as zipf:
                    # 打开并读取特定文件
                    with zipf.open(sample['graph_path'][1]) as file:
                        serialized_dict = file.read()
                        # 反序列化字典对象
                        graph = pickle.loads(serialized_dict)
            except Exception as e:
                graph = None
                print(sample['graph_path'])
                print(e)
        else:
            graph = sample['graph']
    
        # qformer input and target (for pretraining)
        tokenized_input = self.encoder_tokenizer(question, return_attention_mask=False, add_special_tokens=False)
        tokenized_target = self.encoder_tokenizer(label, return_attention_mask=False, add_special_tokens=False)
        
        s = [self.encoder_tokenizer.dec_token_id] + tokenized_input["input_ids"]
        t = [self.encoder_tokenizer.gen_token_id] + tokenized_target["input_ids"] + [self.encoder_tokenizer.eos_token_id]

        qformer_input_ids = torch.LongTensor(s)[:self.max_qformer_length]
        qformer_labels = torch.LongTensor(t)[:self.max_qformer_length]
        
        qformer_input_ids = F.pad(
            qformer_input_ids, (0, self.max_qformer_length - qformer_input_ids.size(0)), value=self.encoder_tokenizer.pad_token_id
        )
        qformer_output_ids = F.pad(
            qformer_labels, (0, self.max_qformer_length - qformer_labels.size(0)), value=self.encoder_tokenizer.pad_token_id
        )
        qformer_labels = F.pad(
            qformer_labels, (0, self.max_qformer_length - qformer_labels.size(0)), value=IGNORE_INDEX
        )        
        
        qformer_input = {
            'graph': graph,
            "input_ids": qformer_input_ids,
            "output_ids": qformer_output_ids,
            "labels": qformer_labels,
        }
        item = {
            "qformer_input": qformer_input,
        }
        if self.sqformer_pretraining:
            return item

        assert '[GRAPH_PAD]' in input_
        if self.num_query_tokens > 0:
            input_ = input_.replace('[GRAPH_PAD]', \
                f'\n\nstruct data representation tokens: {DEFAULT_GRAPH_PAD_TOKEN * self.num_query_tokens}\n\n\n')
        else:
            input_ = input_.replace('[GRAPH_PAD]', '\n\n\n')
                
        # llm input and target
        tokenized_input = self.llm_tokenizer(input_, return_attention_mask=False, add_special_tokens=False)
        tokenized_target = self.llm_tokenizer(label, return_attention_mask=False, add_special_tokens=False)
        s = [self.llm_tokenizer.bos_token_id] + tokenized_input["input_ids"]
        t = tokenized_target["input_ids"] + [self.llm_tokenizer.eos_token_id]

        if self.training:
            input_ids = torch.LongTensor(s + t)[:self.max_seq_length]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:self.max_seq_length]
        else:
            input_ids = torch.LongTensor(s)[:self.max_seq_length]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s))[:self.max_seq_length]

        item.update({
            "input_ids": input_ids,
            "labels": labels,
        })

        return item

    def select(self, idxes):
        self.raw_dataset = self.raw_dataset.select(idxes)
        return self

    def __len__(self):
        return len(self.raw_dataset)


@dataclass
class DataCollatorForGraphSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    llm_tokenizer: transformers.PreTrainedTokenizer
    encoder_tokenizer: transformers.PreTrainedTokenizer

    def _set_llm_padding_side(self):
        # double-underscore name prevents subclasses from (accidentally) overriding the method!
        self.llm_tokenizer.padding_side = "left"

    def _pad_labels(self, batch_labels, padding_side):
        max_label_length = max(len(l) for l in batch_labels)

        for i, labels in enumerate(batch_labels):
            remainder = torch.LongTensor([IGNORE_INDEX] * (max_label_length - len(labels)))
            if padding_side == "right":
                labels = torch.cat([labels, remainder])
            else:
                labels = torch.cat([remainder, labels])
            batch_labels[i] = labels

        return torch.stack(batch_labels)

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        self._set_llm_padding_side()

        # Qformer input
        graphs = [BipartiteData(**instance['qformer_input']["graph"]) for instance in instances]
        graphs = Batch.from_data_list(graphs, follow_batch=['x_s', 'x_t'])
        
        # graphs = merge_graph(graphs)
        # print(graphs['dist_mat'].shape)

        qformer_input_ids = [instance['qformer_input']["input_ids"] for instance in instances]
        qformer_input_ids = self.encoder_tokenizer.pad({"input_ids": qformer_input_ids})["input_ids"]

        qformer_output_ids = [instance['qformer_input']["output_ids"] for instance in instances]
        qformer_output_ids = self.encoder_tokenizer.pad({"input_ids": qformer_input_ids})["input_ids"]

        qformer_labels = [instance['qformer_input']["labels"] for instance in instances]
        qformer_labels = self._pad_labels(qformer_labels, self.encoder_tokenizer.padding_side)

        qformer_inputs = {
            "input_ids": qformer_input_ids,
            "output_ids": qformer_output_ids,
            "labels": qformer_labels,
            "attention_mask": qformer_input_ids.ne(self.encoder_tokenizer.pad_token_id),
            "graphs": graphs,
        }
        batch = {
            'qformer_inputs': qformer_inputs
        } 

        if "input_ids" not in instances[0]:
            # qformer pretraining
            return batch

        # LLM input
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )

        # pad doesn't support pad labels directly
        input_ids = self.llm_tokenizer.pad({"input_ids": input_ids})["input_ids"]
        labels = self._pad_labels(labels, self.llm_tokenizer.padding_side)
        attention_mask = input_ids.ne(self.llm_tokenizer.pad_token_id)
        attention_mask[:, -1] = True


        batch.update({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        })

        return batch


@dataclass
class DataCollatorForGenerating(DataCollatorForGraphSupervisedDataset):
    """Collate examples for generating."""

    def _set_llm_padding_side(self):
        self.llm_tokenizer.padding_side = "left"


if __name__ == "__main__":
    from transformers import AutoTokenizer, set_seed
    from torch.utils.data import DataLoader, Dataset

    set_seed(0)

    dataset_dir = pathlib.Path("data/8Tab_tasks_ori_input_no_inter")

    llm_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_fast=False)
    encoder_tokenizer = AutoTokenizer.from_pretrained(
        "google-bert/bert-base-uncased", use_fast=False)

    graph_pad_token = DEFAULT_GRAPH_PAD_TOKEN
    encoder_tokenizer.add_tokens(
        new_tokens=["[TAB]", "[HEAD]", "[CELL]", "[ROW]", "scinotexp"],
        special_tokens=True,
    )
    llm_tokenizer.add_tokens([graph_pad_token], special_tokens=True)
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

    max_seq_length = 2560
    max_desc_length = 2048
    num_query_tokens = 0
    preprocessing_num_workers = 40
    reprocess = False
    train_dataset = build_instruction_dataset(
        dataset_dir / f"train.pq",
        llm_tokenizer,
        encoder_tokenizer,
        max_seq_length=max_seq_length,
        max_desc_length=max_desc_length,
        num_query_tokens=num_query_tokens,
        preprocessing_num_workers=preprocessing_num_workers,
        reprocess=reprocess,
    )
    val_dataset = build_instruction_dataset(
        dataset_dir / f"val.pq",
        llm_tokenizer,
        encoder_tokenizer,
        max_seq_length=max_seq_length,
        max_desc_length=max_desc_length,
        num_query_tokens=num_query_tokens,
        preprocessing_num_workers=preprocessing_num_workers,
        reprocess=reprocess,
    )
    test_dataset = build_instruction_dataset(
        dataset_dir / f"test.pq",
        llm_tokenizer,
        encoder_tokenizer,
        max_seq_length=max_seq_length,
        max_desc_length=max_desc_length,
        num_query_tokens=num_query_tokens,
        preprocessing_num_workers=preprocessing_num_workers,
        training=False,
        reprocess=reprocess,
    )
    data_collator = DataCollatorForGraphSupervisedDataset(llm_tokenizer)

    loader = DataLoader(val_dataset, 2, collate_fn=data_collator)
    for batch in loader:
        print(batch)
        break

    # from models.hytrel import Encoder
    # from collections import OrderedDict
    # from transformers import AutoConfig
    # @dataclass
    # class OptimizerConfig:
    #     batch_size: int = 256
    #     base_learning_rate: float = 1e-3
    #     weight_decay: float = 0.02
    #     adam_beta1: float = 0.9
    #     adam_beta2: float = 0.98
    #     adam_epsilon: float = 1e-5
    #     lr_scheduler_type: transformers.SchedulerType = "linear"
    #     warmup_step_ratio: float = 0.1
    #     seed: int = 42
    #     optimizer: str = "Adam"
    #     adam_w_mode: bool = True
    #     save_every_n_epochs: int=1
    #     save_top_k: int=1
    #     checkpoint_path: str=''

    #     def __post_init__(self):
    #         if self.optimizer.lower() not in {
    #             "adam",
    #             "fusedadam",
    #             "fusedlamb",
    #             "fusednovograd",
    #         }:
    #             raise KeyError(
    #                 f"The optimizer type should be one of: Adam, FusedAdam, FusedLAMB, FusedNovoGrad. The current value is {self.optimizer}."
    #             )

    #     def get_optimizer(self, optim_groups, learning_rate):
    #         optimizer = self.optimizer.lower()
    #         optim_cls = {
    #             "adam": AdamW if self.adam_w_mode else Adam,
    #         }[optimizer]

    #         args = [optim_groups]
    #         kwargs = {
    #             "lr": learning_rate,
    #             "eps": self.adam_epsilon,
    #             "betas": (self.adam_beta1, self.adam_beta2),
    #         }
    #         if optimizer in {"fusedadam", "fusedlamb"}:
    #             kwargs["adam_w_mode"] = self.adam_w_mode

    #         optimizer = optim_cls(*args, **kwargs)
    #         return optimizer

    # model_config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")
    # model_config.update({'vocab_size': len(encoder_tokenizer), "pre_norm": False, "activation_dropout":0.1, "gated_proj": False})

    # encoder = Encoder(model_config)

    # state_dict = torch.load(
    #     open(
    #         "/home/yaoxu/hypergraph-tabular-lm/checkpoints/electra/epoch=4-step=16345.ckpt/checkpoint/mp_rank_00_model_states.pt",
    #         "rb",
    #     )
    # )

    # new_state_dict = OrderedDict()
    # for k, v in state_dict["module"].items():
    #     if "model" in k:
    #         name = k[13:]  # remove `module.model.`
    #         new_state_dict[name] = v
    # encoder.load_state_dict(new_state_dict, strict=True)

    # encoder.to('cuda')
    # graph = batch['graph']['graph'].to('cuda')
    # x = encoder(batch['graph']['graph'])
