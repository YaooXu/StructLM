from copy import deepcopy
import random
import sys
import time

import numpy as np

sys.path.append("./")

import os

import json
import pickle
import re

import pandas as pd
import torch
from tqdm import tqdm
from utils.utils import df_to_jsonl, load_json, write_jsonl

from collections import defaultdict, deque
from torch_geometric.data import Data
from easydict import EasyDict
from transformers import AutoTokenizer, AutoModel
from multiprocessing import Pool
from datasets import load_dataset
from torch.multiprocessing import Pool, Process, set_start_method
import torch.nn.functional as F


# constants
CAP_TAG = "<caption>"
HEADER_TAG = "<header>"
ROW_TAG = "<row>"

MISSING_CAP_TAG = "[TAB]"
MISSING_CELL_TAG = "[CELL]"
MISSING_HEADER_TAG = "[HEAD]"


class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_index = torch.LongTensor(edge_index) if edge_index else None  # [2, N]
        self.x_s = torch.Tensor(np.array(x_s)) if edge_index else None
        self.x_t = torch.Tensor(np.array(x_t)) if edge_index else None

    def __inc__(self, key, value, *args, **kwargs):
        if key in ["edge_index", "corr_edge_index", "edge_index_corr1", "edge_index_corr2"]:
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


class TableConverter:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.data_args = EasyDict(
            {
                "max_token_length": 64,
                "max_row_length": 30,
                "max_column_length": 20,
                "electra": False,
            }
        )

    # def _tokenize_word(self, word):
    #     # refer to numBERT
    #     number_pattern = re.compile(r"(\d+)\.?(\d*)")  # Matches numbers in decimal form.

    #     def number_repl(matchobj):
    #         """Given a matchobj from number_pattern, it returns a string writing the corresponding number in scientific notation."""
    #         pre = matchobj.group(1).lstrip("0")
    #         post = matchobj.group(2)
    #         if pre and int(pre):
    #             # number is >= 1
    #             exponent = len(pre) - 1
    #         else:
    #             # find number of leading zeros to offset.
    #             exponent = -re.search("(?!0)", post).start() - 1
    #             post = post.lstrip("0")
    #         return (pre + post).rstrip("0") + " scinotexp " + str(exponent)

    #     def apply_scientific_notation(line):
    #         """Convert all numbers in a line to scientific notation."""
    #         res = re.sub(number_pattern, number_repl, line)
    #         return res

    #     # word = apply_scientific_notation(word)
    #     wordpieces = self.tokenizer.tokenize(word)[: self.data_args.max_token_length]

    #     mask = [1 for _ in range(len(wordpieces))]
    #     while len(wordpieces) < self.data_args.max_token_length:
    #         wordpieces.append("[PAD]")
    #         mask.append(0)
    #     return wordpieces, mask

    def _text2table(self, sample):

        smpl = sample.split(HEADER_TAG)
        cap = smpl[0].replace(CAP_TAG, "").strip()
        smpl = smpl[1].split(ROW_TAG)
        headers = [h.strip() for h in smpl[0].strip().split("|")]
        cells = [list(map(lambda x: x.strip(), row.strip().split("|"))) for row in smpl[1:]]
        for row in cells:
            assert len(row) == len(headers)

        return cap, headers, cells

    def _text2graph(self, table, return_dict=False):
        if type(table) is str:
            table = table.replace("col :", "<caption> [TAB] <header>")
            table = re.sub(r"row\s\d+\s:", "<row>", table)
            cap, headers, data = self._text2table(table)
        else:
            if type(table["header"][0]) == list:
                table["header"], table["rows"] = table["header"][0], table["rows"][0]
            cap = ""
            headers, data = table["header"], table["rows"]

        cap = " ".join(cap.split()[: self.data_args.max_token_length])  # filter too long caption
        header = [" ".join(h.split()[: self.data_args.max_token_length]) for h in headers][: self.data_args.max_column_length]
        data = [row[: self.data_args.max_column_length] for row in data[: self.data_args.max_row_length]]

        assert len(header) <= self.data_args.max_column_length
        assert len(data[0]) == len(header)
        assert len(data) <= self.data_args.max_row_length

        wordpieces_xs_all, mask_xs_all = [], []
        wordpieces_xt_all, mask_xt_all = [], []
        nodes, edge_index = [], []

        # caption to hyper-edge (t node)
        cap = f"Table Caption: {cap}"

        tokenized_output = self.tokenizer(
            cap, return_attention_mask=True, padding="max_length", truncation=True, max_length=self.data_args.max_token_length
        )
        wordpieces_xt_all.append(tokenized_output["input_ids"])
        mask_xt_all.append(tokenized_output["attention_mask"])

        # header to hyper-edge (t node)
        for head in header:
            head = f"Table header: {head}"
            tokenized_output = self.tokenizer(
                head,
                return_attention_mask=True,
                padding="max_length",
                truncation=True,
                max_length=self.data_args.max_token_length,
            )
            wordpieces_xt_all.append(tokenized_output["input_ids"])
            mask_xt_all.append(tokenized_output["attention_mask"])

        # row to hyper edge (t node)
        for i in range(len(data)):
            row = f"Row {i+1}"
            tokenized_output = self.tokenizer(
                row,
                return_attention_mask=True,
                padding="max_length",
                truncation=True,
                max_length=self.data_args.max_token_length,
            )
            wordpieces_xt_all.append(tokenized_output["input_ids"])
            mask_xt_all.append(tokenized_output["attention_mask"])

        # cell to nodes (s node)
        for row_i, row in enumerate(data):
            for col_i, word in enumerate(row):
                word = f"Node: {word}"
                tokenized_output = self.tokenizer(
                    word,
                    return_attention_mask=True,
                    padding="max_length",
                    truncation=True,
                    max_length=self.data_args.max_token_length,
                )
                wordpieces_xs_all.append(tokenized_output["input_ids"])
                mask_xs_all.append(tokenized_output["attention_mask"])

                node_id = len(nodes)
                nodes.append(node_id)
                edge_index.append([node_id, 0])  # connect to table-level hyper-edge
                edge_index.append([node_id, col_i + 1])  # # connect to col-level hyper-edge
                edge_index.append([node_id, row_i + 1 + len(header)])  # connect to row-level hyper-edge

        # add label
        # label_ids = torch.zeros((len(header)-1, self.data_args.label_type_num), dtype=torch.float32)
        # assert len(label_ids) == len(labels) == len(header) -1
        col_mask = [0 for i in range(len(wordpieces_xt_all))]
        col_mask[1 : 1 + len(header)] = [1] * len(header)

        # for col_i, lbl in enumerate(labels):
        #     for lbl_i in lbl:
        #         label_ids[col_i, lbl_i] = 1.0
        #         pos_count[lbl_i] += 1

        # xs_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(x) for x in wordpieces_xs_all], dtype=torch.long)
        # xt_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(x) for x in wordpieces_xt_all], dtype=torch.long)

        # with torch.no_grad():
        #     output = llm(
        #         torch.LongTensor(wordpieces_xt_all).to(llm.device), attention_mask=torch.LongTensor(mask_xt_all).to(llm.device)
        #     )

        # check all 0 input
        # xs_tem = torch.count_nonzero(xs_ids, dim=1)
        # xt_tem = torch.count_nonzero(xt_ids, dim=1)
        # assert torch.count_nonzero(xs_tem) == len(xs_tem)
        # assert torch.count_nonzero(xt_tem) == len(xt_tem)
        edge_index = torch.tensor(edge_index, dtype=torch.long).T

        if not return_dict:
            bigraph = BipartiteData(
                edge_index=edge_index,
                x_s=torch.LongTensor(wordpieces_xs_all),
                x_t=torch.LongTensor(wordpieces_xt_all),
                col_mask=col_mask,
                num_nodes=len(wordpieces_xs_all),
                num_hyperedges=len(wordpieces_xt_all),
            )
        else:
            bigraph = dict(
                edge_index=edge_index.tolist(),
                x_s=wordpieces_xs_all,
                x_t=wordpieces_xt_all,
                col_mask=col_mask,
                num_nodes=len(wordpieces_xs_all),
                num_hyperedges=len(wordpieces_xt_all),
            )

        return bigraph


def obtain_samples(process_idx, idxes_to_process):
    new_samples = []
    tasks_to_n = defaultdict(int)
    t1 = time.time()
    for n, idx in enumerate(idxes_to_process):
        if (n + 1) % 100 == 0:
            t2 = time.time()
            print(f"{process_idx}: {n / len(idxes_to_process)}, {t2 - t1}")
            t1 = t2

        sample = samples[idx]
        sample["idx"] = idx

        if "test" in path:
            question = sample["question"] if "question" in sample else sample["statement"]
            struct_data = sample["table"]
            sample["label"] = sample["seq_out"]
            sample["input"] = sample["formatted_input"]

            # if shuffle:
            #     df = pd.DataFrame(sample['table']['rows'], columns=sample['table']['header'])
            #     # shuffle rows
            #     df_shuffled = df.sample(frac=1).reset_index(drop=True)

            #     # shuffle cols
            #     df_shuffled = df_shuffled.iloc[:, :].sample(frac=1, axis=1)
            #     new_rows = ["col : " + " | ".join(df_shuffled.columns)]

            #     for idx, row in df_shuffled.iterrows():
            #         row_str = "row {} : ".format(idx + 1) + " | ".join(map(str, row.values))
            #         new_rows.append(row_str)
            #     new_struct_in = " ".join(new_rows).lower()
            #     new_formatted_input = sample['formatted_input'].replace(sample['struct_in'], new_struct_in)

            #     if len(new_struct_in) != len(sample['struct_in']):
            #         continue

            #     table = new_struct_in
            #     sample['formatted_input'] = new_formatted_input
        else:
            train_data = train_dataset[idx]

            question = sample["input"].split("\n\n")[-1]
            assert (
                question.lower().strip()
                == (train_data["question"] if "question" in train_data else train_data["statement"]).lower().strip()
            )

            struct_data = train_data["table"]

            # table = re.findall(r"table:\n\n([\s\S]*)\n\n\n", sample["input"])[0]
            sys_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n\n\n"
            sample["input"] = sys_prompt + sample["input"] + "\n\n### Response:\n"

        # print(sample['input'])
        # print(question)
        # print(sample['label'])

        try:
            graph = converter._text2graph(struct_data, True)
            if graph:
                sample["graph"] = graph

                if not pretraining:
                    # ft
                    sample["question"] = question
                    new_samples.append(sample)

                    if "test" in path:
                        tasks_to_n[sample["description"]] += 1
                    else:
                        tasks_to_n[sample["task_name"]] += 1
                else:
                    # construct self-supervised data
                    qa_pairs = construct_pretraining_questions(struct_data)
                    for qa_pair in qa_pairs:
                        new_sample = deepcopy(sample)
                        new_sample["question"] = qa_pair[0]
                        new_sample["label"] = qa_pair[1]
                        new_samples.append(new_sample)
        except Exception as e:
            print(e)
            continue

    return new_samples


def random_cell_index(num_rows, num_cols, ignore_first_col=False):
    row_index = random.randint(0, num_rows - 1)

    col_st_idx = 1 if ignore_first_col else 0
    col_index = random.randint(col_st_idx, num_cols - 1)

    return (row_index, col_index)


def sample_ignoring_element(lst, ignore_element):
    filtered_list = [elem for elem in lst if elem != ignore_element]

    sampled_element = random.choice(filtered_list)

    return sampled_element


def construct_pretraining_questions(table, k=10):
    # table: headers : [], rows: [[...], [...], ]
    num_rows = len(table["rows"])
    num_cols = len(table["header"])

    def construct_template1():
        template = 'What\'s the column name of "{node_name}".'

        index = random_cell_index(num_rows, num_cols)
        node_name = table["rows"][index[0]][index[1]]
        col_name = table["header"][index[1]]
        question = template.format(node_name=node_name)

        answer = col_name

        return [(question, answer)]

    def construct_template2():
        template = (
            'In the row where the value of {first_col_name} is "{row_value}", what is the corresponding value of {col_name}?'
        )

        index = random_cell_index(num_rows, num_cols, ignore_first_col=True)
        node_name = table["rows"][index[0]][index[1]]
        col_name = table["header"][index[1]]
        first_col_name = table["header"][0]
        row_value = table["rows"][index[0]][0]

        question = template.format(col_name=col_name, first_col_name=first_col_name, row_value=row_value)
        answer = node_name

        return [(question, answer)]

    def construct_template3():
        template = 'Are "{node_name1}" and "{node_name2}" in the same row?'
        index1 = random_cell_index(num_rows, num_cols)
        node_name1 = table["rows"][index1[0]][index1[1]]

        # same row
        col_idx2 = sample_ignoring_element(list(range(num_cols)), index1[1])
        index2 = (index1[0], col_idx2)
        node_name2 = table["rows"][index2[0]][index2[1]]
        question = template.format(node_name1=node_name1, node_name2=node_name2)
        qa_pair1 = (question, "True")

        # different row
        row_idx3 = sample_ignoring_element(list(range(num_rows)), index2[0])
        index3 = (row_idx3, index2[1])
        node_name3 = table["rows"][index3[0]][index3[1]]
        question = template.format(node_name1=node_name1, node_name2=node_name3)
        qa_pair2 = (question, "False")

        return [qa_pair1, qa_pair2]

    functions = [construct_template1, construct_template2, construct_template3]
    all_qa_pairs = []

    for _ in range(k):
        qa_pairs = random.choice(functions)()
        all_qa_pairs.extend(qa_pairs)

    return all_qa_pairs


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_sentence_embeds(model, tokenizer, input_ids):

    input_ids = torch.LongTensor(input_ids).to(model.device)
    attention_mask = torch.ne(input_ids, tokenizer.pad_token_id).to(input_ids.device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(input_ids, attention_mask=attention_mask)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, attention_mask)

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings


if __name__ == "__main__":

    n_process = 32
    shuffle = False
    pretraining = False
    output_dir = f"data/hytrel/"
    cache_dir = f"/mnt/userdata/StructLM/data/hytrel/cache"

    model_path = "sentence-transformers/all-roberta-large-v1"
    llm = AutoModel.from_pretrained(
        model_path,
        # max_memory={0: "78GiB", 1: "78GiB"},
        device_map="auto",
    )
    llm.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    converter = TableConverter(tokenizer)
    dataset_name = None
    for path, tab_tasks in (
        (
            "data/processed/skginstruct_skgonly.json",
            ["wikitq"],
            # ["fetaqa", "hybridqa", "wikitq", "tabmwp", "wikisql", "tab_fact", "feverous"],
        ),
        (
            "data/processed/skginstruct_test_file_mistral.json",
            ["task: wiki table question"],
            # ["task: tabfact", "task: wiki table question", "task: wikisql"],
        ),
    ):
        all_samples = load_json(path=path)

        tasks_to_samples = defaultdict(list)
        for sample in all_samples:
            if "test" in path:
                tasks_to_samples[sample["description"]].append(sample)
            else:
                tasks_to_samples[sample["task_name"]].append(sample)

        print(list(tasks_to_samples.keys()))

        all_samples = []
        for task in tab_tasks:
            samples = tasks_to_samples[task]

            if "test" not in path:
                train_dataset = load_dataset(f"tasks/{task}.py")["train"]

                print(dataset_name, output_dir)
                assert len(train_dataset) == len(samples)
            else:
                train_dataset = None

            num_samples = len(samples)
            # num_samples = 10
            print(task, num_samples)

            with Pool(processes=n_process) as pool:
                num_samples_in_chunk = num_samples // n_process + 1
                jobs = []
                st = 0
                for i in range(n_process):
                    ed = st + num_samples_in_chunk
                    ed = min(ed, num_samples)
                    jobs.append([i, list(range(st, ed))])
                    st = ed

                results = pool.starmap(obtain_samples, jobs)

            task_samples = []
            for samples in results:
                task_samples.extend(samples)
            print(len(task_samples))

            with torch.no_grad():
                for i, sample in tqdm(enumerate(task_samples)):
                    # replace graph with graph path
                    graph = sample.pop("graph")

                    split = "test" if "test" in path else "train"
                    graph_path = f"{cache_dir}/{task}/{sample['idx']}.pt"
                    sample["graph_path"] = graph_path
                    if os.path.exists(graph_path):
                        continue
                    else:
                        os.makedirs(os.path.dirname(graph_path), exist_ok=True)

                        embedding_s = get_sentence_embeds(llm, tokenizer, graph["x_s"])
                        graph["x_s"] = list(embedding_s.cpu().float().numpy())
                        del embedding_s

                        embedding_t = get_sentence_embeds(llm, tokenizer, graph["x_t"])
                        graph["x_t"] = list(embedding_t.cpu().float().numpy())
                        del embedding_t

                        torch.save(graph, graph_path)

            all_samples.extend(task_samples)

        print(len(all_samples))

        os.makedirs(output_dir, exist_ok=True)
        if "test" in path:
            # if len(tab_tasks) > 1:
            #     random.shuffle(new_samples)

            df = pd.DataFrame(all_samples)

            remain_keys = ["label", "question", "input", "graph_path"]
            sub_df = df[remain_keys]
            sub_df.to_parquet(f"{output_dir}/test.pq", engine="pyarrow", index=False)
            sub_df.to_parquet(f"{output_dir}/val.pq", engine="pyarrow", index=False)

            df_excluded = df.drop(columns=remain_keys)
            df_to_jsonl(df_excluded, f"{output_dir}/ori_test.jsonl")
            df_to_jsonl(df_excluded, f"{output_dir}/ori_val.jsonl")
        else:
            df = pd.DataFrame(all_samples)
            df.to_parquet(f"{output_dir}/train.pq", engine="pyarrow", index=False)

        print("done")
