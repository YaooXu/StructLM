import random
import sys
import os

import json
import pickle
import re

import pandas as pd
import torch
from tqdm import tqdm
from utils.utils import load_json

from collections import defaultdict
from torch_geometric.data import Data
from easydict import EasyDict
from transformers import AutoTokenizer

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

        word = apply_scientific_notation(word)
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

    def _text2graph(self, table_str, return_dict=False):
        table_str = table_str.replace("col :", "<caption> [TAB] <header>")
        table_str = re.sub(r"row\s\d+\s:", "<row>", table_str)
        try:
            cap, headers, data = self._text2table(table_str)
        except:
            print("Fail to parser the table...")
            # cap, headers, data = self._text2table(table_str)
            return None

        cap = " ".join(cap.split()[: self.data_args.max_token_length])  # filter too long caption
        header = [" ".join(h.split()[: self.data_args.max_token_length]) for h in headers][
            : self.data_args.max_column_length
        ]
        data = [
            row[: self.data_args.max_column_length] for row in data[: self.data_args.max_row_length]
        ]

        assert len(header) <= self.data_args.max_column_length
        assert len(data[0]) == len(header)
        assert len(data) <= self.data_args.max_row_length

        wordpieces_xs_all, mask_xs_all = [], []
        wordpieces_xt_all, mask_xt_all = [], []
        nodes, edge_index = [], []

        # caption to hyper-edge (t node)
        if not cap:
            wordpieces = ["[TAB]"] + ["[PAD]" for _ in range(self.data_args.max_token_length - 1)]
            mask = [1] + [0 for _ in range(self.data_args.max_token_length - 1)]
            wordpieces_xt_all.append(wordpieces)
            mask_xt_all.append(mask)
        else:
            wordpieces, mask = self._tokenize_word(cap)
            wordpieces_xt_all.append(wordpieces)
            mask_xt_all.append(mask)

        # header to hyper-edge (t node)
        for head in header:
            if not head:
                wordpieces = ["[HEAD]"] + [
                    "[PAD]" for _ in range(self.data_args.max_token_length - 1)
                ]
                mask = [1] + [0 for _ in range(self.data_args.max_token_length - 1)]
                wordpieces_xt_all.append(wordpieces)
                mask_xt_all.append(mask)
            else:
                wordpieces, mask = self._tokenize_word(head)
                wordpieces_xt_all.append(wordpieces)
                mask_xt_all.append(mask)

        # row to hyper edge (t node)
        for i in range(len(data)):
            wordpieces = ["[ROW]"] + ["[PAD]" for _ in range(self.data_args.max_token_length - 1)]
            mask = [1] + [0 for _ in range(self.data_args.max_token_length - 1)]
            wordpieces_xt_all.append(wordpieces)
            mask_xt_all.append(mask)

        # cell to nodes (s node)
        for row_i, row in enumerate(data):
            for col_i, word in enumerate(row):
                if not word:
                    wordpieces = ["[CELL]"] + [
                        "[PAD]" for _ in range(self.data_args.max_token_length - 1)
                    ]
                    mask = [1] + [0 for _ in range(self.data_args.max_token_length - 1)]
                else:
                    word = " ".join(word.split()[: self.data_args.max_token_length])
                    wordpieces, mask = self._tokenize_word(word)
                wordpieces_xs_all.append(wordpieces)
                mask_xs_all.append(mask)
                node_id = len(nodes)
                nodes.append(node_id)
                edge_index.append([node_id, 0])  # connect to table-level hyper-edge
                edge_index.append([node_id, col_i + 1])  # # connect to col-level hyper-edge
                edge_index.append(
                    [node_id, row_i + 1 + len(header)]
                )  # connect to row-level hyper-edge

        # add label
        # label_ids = torch.zeros((len(header)-1, self.data_args.label_type_num), dtype=torch.float32)
        # assert len(label_ids) == len(labels) == len(header) -1
        col_mask = [0 for i in range(len(wordpieces_xt_all))]
        col_mask[1: 1 + len(header)] = [1] * len(header)

        # for col_i, lbl in enumerate(labels):
        #     for lbl_i in lbl:
        #         label_ids[col_i, lbl_i] = 1.0
        #         pos_count[lbl_i] += 1

        xs_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(x) for x in wordpieces_xs_all], dtype=torch.long
        )
        xt_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(x) for x in wordpieces_xt_all], dtype=torch.long
        )

        # check all 0 input
        xs_tem = torch.count_nonzero(xs_ids, dim=1)
        xt_tem = torch.count_nonzero(xt_ids, dim=1)
        assert torch.count_nonzero(xs_tem) == len(xs_tem)
        assert torch.count_nonzero(xt_tem) == len(xt_tem)
        edge_index = torch.tensor(edge_index, dtype=torch.long).T

        if not return_dict:
            bigraph = BipartiteData(
                edge_index=edge_index,
                x_s=xs_ids,
                x_t=xt_ids,
                col_mask=col_mask,
                num_nodes=len(xs_ids),
                num_hyperedges=len(xt_ids),
            )
        else:
            bigraph = dict(
                edge_index=edge_index,
                x_s=xs_ids,
                x_t=xt_ids,
                col_mask=col_mask,
                num_nodes=len(xs_ids),
                num_hyperedges=len(xt_ids),
            )

        return bigraph


if __name__ == "__main__":

    path = "data/processed/skginstruct.json"
    tab_tasks = ['tabmwp', 'hybridqa', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa']
    tab_tasks = ['tabmwp']

    # path = "data/processed/skginstruct_test_file_13b_34b.json"
    # path = "data/processed/skginstruct_test_file_7b.json"
    # tab_tasks = ['task: tabmwp', 'task: hybridqa', 'task: tabfact',
    #              'task: wiki table question', 'task: wikisql', 'task: fetaqa']
    # tab_tasks = ['task: wiki table question']
    all_samples = load_json(path)

    tasks_to_samples = defaultdict(list)
    for sample in all_samples:
        if "test" in path:
            tasks_to_samples[sample["description"]].append(sample)
        else:
            tasks_to_samples[sample["task_name"]].append(sample)

    samples = []
    for task in tab_tasks:
        samples.extend(tasks_to_samples[task])

    print(len(samples))

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    new_tokens = ["[TAB]", "[HEAD]", "[CELL]", "[ROW]", "scinotexp"]
    tokenizer.add_tokens(new_tokens, special_tokens=True)

    converter = TableConverter(tokenizer)

    new_samples = []
    for sample in tqdm(samples):
        if "test" in path:
            question = sample["question"] if 'question' in sample else sample['statement']
            table = sample["struct_in"]
            sample['label'] = sample['seq_out']
            sample['inst'] = re.findall(
                r"(<<SYS>>[\s\S]*table:\n\n)", sample["formatted_input"])[0]
        else:
            question = re.findall(
                r"(question|statement):\n\n(.*)", sample["input"])[0][1]
            table = re.findall(r"table:\n\n([\s\S]*)\n\n\n", sample["input"])[0]
            sys_prompt = '<<SYS>>\n' + sample['sys_prompt'] + '\n<</SYS>>\n\n'
            sample["inst"] = sys_prompt + re.findall(
                r"([\s\S]*table:\n\n)", sample["input"])[0]

        # print(sample['inst'])

        sample["question"] = question

        graph = converter._text2graph(table)
        if graph:
            sample["struct_in"] = table
            new_samples.append(sample)

    print(len(new_samples), len(samples))
    if "test" in path:
        with open(f"data/TAB_SYS_PROPMT/test.jsonl", "w") as f:
            for sample in new_samples:
                f.write(json.dumps(sample) + "\n")
    else:
        random.shuffle(new_samples)
        num_train_samples = int(len(new_samples) * 0.95)
        train, val = (
            new_samples[:num_train_samples],
            new_samples[num_train_samples:],
        )

        for dataset, name in zip([train, val], ["train", "val"]):
            with open(f"data/TAB_SYS_PROPMT/{name}.jsonl", "w") as f:
                for sample in dataset:
                    f.write(json.dumps(sample) + "\n")

    print("done")
