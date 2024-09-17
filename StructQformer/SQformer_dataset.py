import sys
sys.path.append('./')

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
from StructQformer.utils.data import BipartiteData, TableConverter
from utils.utils import load_json

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
            key_to_array[key] = [
                torch.nn.utils.rnn.pad_sequence(node_token_ids, batch_first=True)
                for node_token_ids in lists
            ]
        elif key in ["node_types", "node_attrs", "graph_attention_mask"]:
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
    bert_tokenizer: transformers.PreTrainedTokenizer,
    max_seq_length: int = 2560,
    max_desc_length: int = 2048,
    data_cache_dir=None,
    preprocessing_num_workers=32,
    num_query_tokens=10,
    training=True,
    shuffle_desc=True,
    reprocess=False,
):
    """_summary_

    Args:
        data_path (Union[List[str], str]): _description_
        llm_tokenizer (transformers.PreTrainedTokenizer): tokenizer for down-stream LLM
        bert_tokenizer (transformers.PreTrainedTokenizer): tokenizer for g-former
        max_seq_length (int, optional): _description_. Defaults to 256.
        max_desc_length: Defaults to 0
        data_cache_dir (_type_, optional): _description_. Defaults to None.
        preprocessing_num_workers (int, optional): _description_. Defaults to 10.
    """

    assert max_seq_length > max_desc_length

    converter = TableConverter(bert_tokenizer)

    def tokenization(examples):
        sources = []
        targets = []
        questions = []
        insts = []
        struct_ins = []
        bi_graphs = []

        for label, question, inst, struct_in in zip(
            examples["label"], examples["question"], examples["inst"], examples["struct_in"]
        ):  
            if num_query_tokens > 0:
                source = f"\n\ntable representation tokens: {DEFAULT_GRAPH_PAD_TOKEN * num_query_tokens}\n\n\nquestion:\n\n{question}"
            else:
                source = f"\n\n\nquestion:\n\n{question}"
            source += "\n\n### Response:\n"

            target = f"{label}{llm_tokenizer.eos_token}"

            sources.append(source)
            targets.append(target)
            questions.append(question)
            bi_graphs.append(converter._text2graph(struct_in, return_dict=True))

            # if shuffle_desc:
            #     node_texts = desc.split("\n")
            #     random.shuffle(node_texts)
            #     desc = "\n".join(node_texts)

            insts.append(inst)
            struct_ins.append(struct_in)

        # add_special_tokens=False: not add <s>
        tokenized_insts = llm_tokenizer(
            insts, return_attention_mask=False, add_special_tokens=False
        )
        tokenized_struct_ins = llm_tokenizer(
            struct_ins, return_attention_mask=False, add_special_tokens=False
        )
        tokenized_sources = llm_tokenizer(
            sources, return_attention_mask=False, add_special_tokens=False
        )
        tokenized_targets = llm_tokenizer(
            targets, return_attention_mask=False, add_special_tokens=False
        )
        tokenized_questions = bert_tokenizer(questions)

        all_input_ids = []
        all_labels = []
        all_question_ids = []

        # lens = []

        for inst, struct_in, s, t, q in zip(
            tokenized_insts["input_ids"],
            tokenized_struct_ins["input_ids"],
            tokenized_sources["input_ids"],
            tokenized_targets["input_ids"],
            tokenized_questions["input_ids"],
        ):
            s = [llm_tokenizer.bos_token_id] + inst + struct_in[:max_desc_length] + s
            if training:
                input_ids = torch.LongTensor(s + t)[:max_seq_length]
                labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            else:
                input_ids = torch.LongTensor(s)[:max_seq_length]
                labels = torch.LongTensor([IGNORE_INDEX] * len(s))[:max_seq_length]

            question_ids = torch.LongTensor(q)
            assert len(input_ids) == len(labels)

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_question_ids.append(question_ids)

            # lens.append(len(s + t))

        # hist, bins = np.histogram(lens, bins=10)
        # for i in range(len(hist)):
        #     print(f"{int(bins[i])} -  {int(bins[i+1])}: {hist[i]}")

        results = {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "question_ids": all_question_ids,
            "graph": bi_graphs,
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
            os.path.basename(file).split(".")[0] + \
                f"{max_desc_length}_{max_seq_length}_{num_query_tokens}_{llm_tokenizer.name_or_path.split('/')[-1]}",
        )
        os.makedirs(cache_path, exist_ok=True)
        try:
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info(f"training datasets-{file} has been loaded from disk")
        except Exception as e:
            raw_dataset = load_dataset("json", data_files=file)
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


@dataclass
class DataCollatorForGraphSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    llm_tokenizer: transformers.PreTrainedTokenizer

    def _set_llm_padding_side(self):
        # double-underscore name prevents subclasses from (accidentally) overriding the method!
        self.llm_tokenizer.padding_side = "right"

    def _pad_labels(self, batch_labels):
        max_label_length = max(len(l) for l in batch_labels)
        padding_side = self.llm_tokenizer.padding_side

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

        graphs = [BipartiteData(**instance["graph"]) for instance in instances]

        graphs = Batch.from_data_list(graphs)

        # Qformer input
        question_ids = [instance["question_ids"] for instance in instances]
        question_ids = torch.nn.utils.rnn.pad_sequence(question_ids, batch_first=True)
        batch_graph = {
            "question_input_ids": question_ids,
            "question_attention_mask": question_ids.ne(0),
            "graph": graphs,
        }

        # LLM input
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )

        # pad doesn't support pad labels directly
        input_ids = self.llm_tokenizer.pad({"input_ids": input_ids})["input_ids"]
        labels = self._pad_labels(labels)
        attention_mask = input_ids.ne(self.llm_tokenizer.pad_token_id)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        batch["graph"] = batch_graph

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

    dataset_dir = pathlib.Path("data/WTQ_Mistral")

    llm_tokenizer = AutoTokenizer.from_pretrained("TIGER-Lab/StructLM-7B-Mistral", use_fast=False)
    bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", use_fast=False)

    graph_pad_token = DEFAULT_GRAPH_PAD_TOKEN
    bert_tokenizer.add_tokens(
        ["[TAB]", "[HEAD]", "[CELL]", "[ROW]", "scinotexp"],
        special_tokens=True,
    )
    llm_tokenizer.add_tokens([graph_pad_token], special_tokens=True)
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

    max_seq_length = 2560
    max_desc_length = 2048
    num_query_tokens = 10
    preprocessing_num_workers = 32
    reprocess = True
    train_dataset = build_instruction_dataset(
        dataset_dir / f"train.jsonl",
        llm_tokenizer,
        bert_tokenizer,
        max_seq_length=max_seq_length,
        max_desc_length=max_desc_length,
        num_query_tokens=num_query_tokens,
        preprocessing_num_workers=preprocessing_num_workers,
        reprocess=reprocess,
    )
    val_dataset = build_instruction_dataset(
        dataset_dir / f"val.jsonl",
        llm_tokenizer,
        bert_tokenizer,
        max_seq_length=max_seq_length,
        max_desc_length=max_desc_length,
        num_query_tokens=num_query_tokens,
        preprocessing_num_workers=preprocessing_num_workers,
        reprocess=reprocess,
    )
    test_dataset = build_instruction_dataset(
        dataset_dir / f"test.jsonl",
        llm_tokenizer,
        bert_tokenizer,
        max_seq_length=max_seq_length,
        max_desc_length=max_desc_length,
        num_query_tokens=num_query_tokens,
        preprocessing_num_workers=preprocessing_num_workers,
        training=False,
        reprocess=reprocess,
    )
    data_collator = DataCollatorForGraphSupervisedDataset(llm_tokenizer)

    loader = DataLoader(train_dataset, 2, collate_fn=data_collator)
    for batch in loader:
        print(llm_tokenizer.decode(batch['input_ids'][0]))
        break

    # from StructQformer.models.hytrel import Encoder
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
    # model_config.update({'vocab_size': len(bert_tokenizer), "pre_norm": False, "activation_dropout":0.1, "gated_proj": False})

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
