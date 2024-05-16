import random
import sys
import time

import numpy as np
sys.path.append('./')

import os

import json
import pickle
import re

import pandas as pd
import torch
from tqdm import tqdm
from utils.utils import load_json, write_jsonl

from collections import defaultdict, deque
from torch_geometric.data import Data
from easydict import EasyDict
from transformers import AutoTokenizer
from multiprocessing import Pool

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
        self.edge_index = edge_index  # [2, N]
        self.x_s = x_s
        self.x_t = x_t

    def __inc__(self, key, value, *args, **kwargs):
        if key in ["edge_index", "corr_edge_index", "edge_index_corr1", "edge_index_corr2"]:
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


def construct_graph_from_edge_index(edge_index):
    graph = defaultdict(list)

    for h, t in zip(edge_index[0], edge_index[1]):
        graph[h].append(t)
        graph[t].append(h)

    return graph


def bfs(source, graph, num_nodes):
    dist = [-1] * num_nodes

    visited = {source}
    dist[source] = 0

    queue = deque([source])

    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v not in visited:
                queue.append(v)
                visited.add(v)
                dist[v] = dist[u] + 1

    return dist


def _get_dist_mat(num_nodes, edge_index):
    # # get shortest distance between two nodes

    graph = construct_graph_from_edge_index(edge_index)

    dist_mat = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        dist = bfs(i, graph, num_nodes)
        dist_mat[i] = dist

    return dist_mat


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

    def _tokenize_word(self, word):
        # refer to numBERT
        number_pattern = re.compile(r"(\d+)\.?(\d*)")  # Matches numbers in decimal form.

        def number_repl(matchobj):
            """Given a matchobj from number_pattern, it returns a string writing the corresponding number in scientific notation."""
            pre = matchobj.group(1).lstrip("0")
            post = matchobj.group(2)
            if pre and int(pre):
                # number is >= 1
                exponent = len(pre) - 1
            else:
                # find number of leading zeros to offset.
                exponent = -re.search("(?!0)", post).start() - 1
                post = post.lstrip("0")
            return (pre + post).rstrip("0") + " scinotexp " + str(exponent)

        def apply_scientific_notation(line):
            """Convert all numbers in a line to scientific notation."""
            res = re.sub(number_pattern, number_repl, line)
            return res

        # word = apply_scientific_notation(word)
        wordpieces = self.tokenizer.tokenize(word)[: self.data_args.max_token_length]

        mask = [1 for _ in range(len(wordpieces))]
        while len(wordpieces) < self.data_args.max_token_length:
            wordpieces.append("[PAD]")
            mask.append(0)
        return wordpieces, mask

    def _text2table(self, sample):

        smpl = sample.split(HEADER_TAG)
        cap = smpl[0].replace(CAP_TAG, "").strip()
        smpl = smpl[1].split(ROW_TAG)
        headers = [h.strip() for h in smpl[0].strip().split("|")]
        cells = [list(map(lambda x: x.strip(), row.strip().split("|"))) for row in smpl[1:]]
        for row in cells:
            assert len(row) == len(headers)

        return cap, headers, cells

    def _text2graph(self, table_str, get_dist_mat=True):

        table_str = table_str.replace("col :", "<caption> [TAB] <header>")
        table_str = re.sub(r"row\s\d+\s:", "<row>", table_str)

        try:
            cap, headers, data = self._text2table(table_str)

            # filter too long caption
            cap = " ".join(cap.split()[: self.data_args.max_token_length])
            header = [" ".join(h.split()[: self.data_args.max_token_length]) for h in headers][
                : self.data_args.max_column_length
            ]
            data = [
                row[: self.data_args.max_column_length] for row in data[: self.data_args.max_row_length]
            ]

            assert len(header) <= self.data_args.max_column_length
            assert len(data[0]) == len(header)
            assert len(data) <= self.data_args.max_row_length

            wordpieces_all, mask_all = [], []
            node_types, edge_index = [], []

            for head in header:
                if not head:
                    wordpieces = ["[HEAD]"] + [
                        "[PAD]" for _ in range(self.data_args.max_token_length - 1)
                    ]
                    mask = [1] + [0 for _ in range(self.data_args.max_token_length - 1)]
                    wordpieces_all.append(wordpieces)
                    mask_all.append(mask)
                else:
                    wordpieces, mask = self._tokenize_word(head)
                    wordpieces_all.append(wordpieces)
                    mask_all.append(mask)
                node_types.append(0)

            for row_i, row in enumerate(data):

                # ROW node
                row_node_id = len(wordpieces_all)
                wordpieces = ["[ROW]"] + \
                    ["[PAD]" for _ in range(self.data_args.max_token_length - 1)]
                mask = [1] + [0 for _ in range(self.data_args.max_token_length - 1)]
                wordpieces_all.append(wordpieces)
                mask_all.append(mask)
                node_types.append(1)

                for col_i, word in enumerate(row):
                    col_node_id = col_i

                    if word.strip() in ['-', '']:
                        wordpieces = ["[CELL]"] + [
                            "[PAD]" for _ in range(self.data_args.max_token_length - 1)
                        ]
                        mask = [1] + [0 for _ in range(self.data_args.max_token_length - 1)]
                    else:
                        word = " ".join(word.split()[: self.data_args.max_token_length])
                        wordpieces, mask = self._tokenize_word(word)

                    node_id = len(wordpieces_all)
                    wordpieces_all.append(wordpieces)
                    mask_all.append(mask)
                    node_types.append(2)

                    edge_index.append([node_id, col_node_id])
                    edge_index.append([col_node_id, node_id])

                    edge_index.append([node_id, row_node_id])
                    edge_index.append([row_node_id, node_id])

            node_token_ids = torch.tensor(
                [self.tokenizer.convert_tokens_to_ids(x) for x in wordpieces_all], dtype=torch.long
            )
            edge_index = np.array(edge_index).T
            node_types = np.array(node_types)
            graph = {
                "edge_index": edge_index,
                'node_token_ids': node_token_ids,
                "node_types": node_types,
            }
            if get_dist_mat:
                dist_mat = _get_dist_mat(len(wordpieces_all), edge_index)
                graph['dist_mat'] = dist_mat

            return graph
        except:
            print("Fail to parser the table...")
            # cap, headers, data = self._text2table(table_str)
            return None


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

        if "test" in path:
            question = sample["question"] if 'question' in sample else sample['statement']
            table = sample["struct_in"]
            sample['label'] = sample['seq_out']
            sample['inst'] = re.findall(
                r"([\s\S]*table:\n\n)", sample["formatted_input"])[0]
        else:
            question = re.findall(
                r"(question|statement):\n\n(.*)", sample["input"])[0][1]
            table = re.findall(r"table:\n\n([\s\S]*)\n\n\n", sample["input"])[0]
            # sys_prompt = '<<SYS>>\n' + sample['sys_prompt'] + '\n<</SYS>>\n\n'
            sys_prompt = 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n\n\n'
            sample["inst"] = sys_prompt + re.findall(
                r"([\s\S]*table:\n\n)", sample["input"])[0][0]

        print(sample['inst'])
        print(table)
        print(question)
        print(sample['label'])

        sample["question"] = question

        graph = converter._text2graph(table, True)
        if graph:
            sample["struct_in"] = table
            new_samples.append(sample)

            if "test" in path:
                tasks_to_n[sample["description"]] += 1
            else:
                tasks_to_n[sample["task_name"]] += 1

    print(tasks_to_n)

    return new_samples


if __name__ == "__main__":

    # path = "data/processed/skginstruct_skgonly.json"
    # tab_tasks = ['tab_fact', 'wikitq', 'wikisql', 'tabmwp', 'fetaqa']
    # tab_tasks = ['wikitq']

    path = "data/processed/skginstruct_test_file_mistral.json"
    tab_tasks = ['task: tabfact', 'task: wiki table question', 'task: wikisql']
    tab_tasks = ['task: wiki table question']
    all_samples = load_json(path)

    tasks_to_samples = defaultdict(list)
    for sample in all_samples:
        if "test" in path:
            tasks_to_samples[sample["description"]].append(sample)
        else:
            tasks_to_samples[sample["task_name"]].append(sample)
    print(list(tasks_to_samples.keys()))

    samples = []
    for task in tab_tasks:
        samples.extend(tasks_to_samples[task])

    num_samples = len(samples)
    print(num_samples)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    new_tokens = ["[HEAD]", "[CELL]", "[ROW]"]
    tokenizer.add_tokens(new_tokens, special_tokens=True)

    converter = TableConverter(tokenizer)

    n_process = 1
    with Pool(processes=n_process) as pool:
        num_samples_in_chunk = num_samples // n_process
        jobs = []
        st = 0
        for i in range(n_process):
            ed = st + num_samples_in_chunk
            ed = min(ed, num_samples)
            jobs.append([i, list(range(st, ed))])
            st = ed

        results = pool.starmap(obtain_samples, jobs)

    new_samples = []
    for samples in results:
        new_samples.extend(samples)

    random.shuffle(new_samples)
    if "test" in path:
        write_jsonl(f"data/WTQ_Mistral_Tabert/ori_test.jsonl", new_samples)
        write_jsonl(f"data/WTQ_Mistral_Tabert/ori_val.jsonl", new_samples[:1000])

        remain_keys = ['label', 'question', 'inst', 'struct_in']
        samples = []
        for sample in new_samples:
            samples.append({k: sample[k] for k in remain_keys})
        write_jsonl(f"data/WTQ_Mistral_Tabert/test.jsonl", samples)
        write_jsonl(f"data/WTQ_Mistral_Tabert/val.jsonl", samples[:1000])
    else:
        train = new_samples

        for dataset, name in zip([train], ["train"]):
            with open(f"data/WTQ_Mistral_Tabert/{name}.jsonl", "w") as f:
                for sample in dataset:
                    f.write(json.dumps(sample) + "\n")

    print("done")
