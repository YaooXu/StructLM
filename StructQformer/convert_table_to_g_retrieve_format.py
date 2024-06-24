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
from utils.utils import df_to_jsonl, load_json, write_jsonl

from collections import defaultdict, deque
from torch_geometric.data import Data
from easydict import EasyDict
from transformers import AutoTokenizer
from multiprocessing import Pool
from datasets import load_dataset

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


def cutoff(word, n=64):
    return ' '.join(word.split()[:n])


class StructDataConverter:
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
            wordpieces.append(self.tokenizer.pad_token)
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

    def _table2graph(self, table):
        try:
            if type(table) is str:
                table = table.replace("col :", "<caption> [TAB] <header>")
                table = re.sub(r"row\s\d+\s:", "<row>", table)
                cap, headers, data = self._text2table(table)
            else:
                if type(table['header'][0]) == list:
                    table['header'], table['rows'] = table['header'][0], table['rows'][0]
                cap = ''
                headers, data = table['header'], table['rows']

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

            node_lines = []
            edge_lines = []

            for head in header:
                if not head:
                    wordpieces = ["[HEAD]"] + [
                        self.tokenizer.pad_token for _ in range(self.data_args.max_token_length - 1)
                    ]
                    mask = [1] + [0 for _ in range(self.data_args.max_token_length - 1)]
                else:
                    wordpieces, mask = self._tokenize_word(head)
                    if wordpieces == [self.tokenizer.pad_token] * self.data_args.max_token_length:
                        wordpieces[0] = '[HEAD]'
                        
                node_lines.append(f'{len(wordpieces_all)}, {cutoff(head)}')
                wordpieces_all.append(wordpieces)
                mask_all.append(mask)
                node_types.append(0)

            for row_i, row in enumerate(data):

                # ROW node
                row_node_id = len(wordpieces_all)
                node_lines.append(f'{row_node_id}, ROW {row_i}')

                wordpieces = ["[ROW]"] + \
                    [self.tokenizer.pad_token for _ in range(self.data_args.max_token_length - 1)]
                mask = [1] + [0 for _ in range(self.data_args.max_token_length - 1)]
                wordpieces_all.append(wordpieces)
                mask_all.append(mask)
                node_types.append(1)

                for col_i, word in enumerate(row):
                    col_node_id = col_i

                    word = word.strip()
                    if word in ['-', '']:
                        wordpieces = ["[CELL]"] + [
                            self.tokenizer.pad_token for _ in range(self.data_args.max_token_length - 1)
                        ]
                        mask = [1] + [0 for _ in range(self.data_args.max_token_length - 1)]
                    else:
                        word = " ".join(word.split()[: self.data_args.max_token_length])
                        wordpieces, mask = self._tokenize_word(word)

                    # some special unicode may lead to this, WTF
                    if wordpieces == [self.tokenizer.pad_token] * self.data_args.max_token_length:
                        wordpieces[0] = '[CELL]'

                    node_id = len(wordpieces_all)
                    node_lines.append(f'{node_id}, {cutoff(word)}')

                    wordpieces_all.append(wordpieces)
                    mask_all.append(mask)
                    node_types.append(2)

                    edge_index.append([node_id, col_node_id])
                    edge_index.append([col_node_id, node_id])
                    edge_lines.append(f'{node_id}, column_name, {col_node_id}')

                    edge_index.append([node_id, row_node_id])
                    edge_index.append([row_node_id, node_id])
                    edge_lines.append(f'{node_id}, row_name, {row_node_id}')

            node_token_ids = torch.tensor(
                [self.tokenizer.convert_tokens_to_ids(x) for x in wordpieces_all], dtype=torch.long
            )
            edge_index = np.array(edge_index).T
            node_types = np.array(node_types)
            dist_mat = _get_dist_mat(len(wordpieces_all), edge_index)

            # check all 0 input
            xs_tem = torch.count_nonzero(node_token_ids - self.tokenizer.pad_token_id, dim=1)
            assert torch.count_nonzero(xs_tem) == len(xs_tem)

            graph = {
                "edge_index": edge_index.tolist(),
                'node_token_ids': node_token_ids.tolist(),
                "node_types": node_types.tolist(),
                "dist_mat": dist_mat.tolist()
            }

            text_graph = 'node_id, node_attr\n' + \
                '\n'.join(node_lines)
                # '\nsrc, edge_attr, dst\n' + \
                # '\n'.join(edge_lines)
            return graph, text_graph

        except Exception as e:
            print(e)
            print("Fail to parser the table...")
            return None, None

    def _kg2graph(self, tuples):
        wordpieces_all = []
        name_to_id = {}
        edge_index = []
        node_types = []

        try:
            for tup in tuples:
                h, r, t = tup
                if h not in name_to_id:
                    name_to_id[h] = len(name_to_id)
                    wordpieces_all.append(self._tokenize_word(h)[0])
                    node_types.append(0)

                name_to_id[r] = len(name_to_id)
                wordpieces_all.append(self._tokenize_word(r)[0])
                node_types.append(1)

                if t not in name_to_id:
                    name_to_id[t] = len(name_to_id)
                    wordpieces_all.append(self._tokenize_word(t)[0])
                    node_types.append(0)

                h_idx, r_idx, t_idx = name_to_id[h], name_to_id[r], name_to_id[t]

                edge_index.append([h_idx, r_idx])
                edge_index.append([r_idx, h_idx])

                edge_index.append([r_idx, t_idx])
                edge_index.append([t_idx, r_idx])

            node_token_ids = torch.tensor(
                [self.tokenizer.convert_tokens_to_ids(x) for x in wordpieces_all], dtype=torch.long
            )
            edge_index = np.array(edge_index).T
            node_types = np.array(node_types)
            dist_mat = _get_dist_mat(len(wordpieces_all), edge_index)

            # check all 0 input
            xs_tem = torch.count_nonzero(node_token_ids - self.tokenizer.pad_token_id, dim=1)
            assert torch.count_nonzero(xs_tem) == len(xs_tem)

            graph = {
                "edge_index": edge_index.tolist(),
                'node_token_ids': node_token_ids.tolist(),
                "node_types": node_types.tolist(),
                "dist_mat": dist_mat.tolist()
            }

            return graph
        except Exception as e:
            print(e)
            print("Fail to parser the kg...")
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
            struct_data_key = 'table' if 'table' in sample else 'kg_tuples'

            question = sample['question'] if 'question' in sample else sample['statement']
            struct_data = sample[struct_data_key]
            sample['label'] = sample['seq_out']
            sample["input"] = sample["formatted_input"]
            # print(sample["formatted_input"])

        else:
            train_data = train_dataset[idx]
            struct_data_key = 'table' if 'table' in train_data else 'kg_tuples'

            question = sample["input"].split('\n\n')[-1]
            assert question.lower().strip() == (
                train_data['question'] if 'question' in train_data else train_data['statement']).lower().strip()

            struct_data = train_data[struct_data_key]

            # table = re.findall(r"table:\n\n([\s\S]*)\n\n\n", sample["input"])[0]
            sys_prompt = 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n\n\n'
            sample["input"] = sys_prompt + sample["input"] + "\n\n### Response:\n"
            # print(sample['input'])

        sample["question"] = question

        if struct_data_key == 'table':
            graph, text = converter._table2graph(struct_data)
        elif struct_data_key == 'kg_tuples':
            graph = converter._kg2graph(struct_data)

        sample["input"] = text + f'\n\nQuestion: {question}'
        
        if graph:
            sample['graph'] = graph
            # sample["struct_in"] = table
            new_samples.append(sample)

            if "test" in path:
                tasks_to_n[sample["description"]] += 1
            else:
                tasks_to_n[sample["task_name"]] += 1

    print(tasks_to_n)

    return new_samples


if __name__ == "__main__":

    output_dir = 'WTQ'
    output_dir = f'data_g_retrieve/roberta/{output_dir}'
    os.makedirs(output_dir, exist_ok=True)
    n_process = 32

    # path = "data/processed/skginstruct_skgonly.json"
    # tab_tasks = ['fetaqa', 'hybridqa', 'wikitq', 'tabmwp', 'mmqa', 'wikisql', 'tab_fact', 'feverous']
    # tab_tasks = ['wikitq']

    # path = "data/processed/skginstruct_test_file_mistral.json"
    # tab_tasks = ['task: tabfact', 'task: wiki table question', 'task: wikisql', 'task: hybridqa', 'task: compwebq']
    # tab_tasks = ['task: wiki table question']

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    # tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    new_tokens = ["[TAB]", "[HEAD]", "[CELL]", "[ROW]", "scinotexp"]
    tokenizer.add_tokens(new_tokens, special_tokens=True)

    converter = StructDataConverter(tokenizer)

    # for path, tab_tasks in zip(["data/processed/skginstruct_skgonly.json", "data/processed/skginstruct_test_file_mistral.json"],
    #                            [['fetaqa', 'hybridqa', 'wikitq', 'tabmwp', 'wikisql', 'tab_fact', 'feverous'], ['task: wiki table question']]):
    # for path, tab_tasks in zip(["data/processed/skginstruct_skgonly.json", "data/processed/skginstruct_test_file_mistral.json"],
    #                            [['fetaqa', 'hybridqa', 'wikitq', 'tabmwp', 'mmqa', 'wikisql', 'tab_fact', 'feverous', 'compwebq'],
    #                             ['task: tabfact', 'task: wiki table question', 'task: wikisql', 'task: compwebq']]):
    for path, tab_tasks in zip(["data/processed/skginstruct_skgonly.json", "data/processed/skginstruct_test_file_mistral.json"],
                               [['wikitq'], ['task: wiki table question']]):
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

            if 'test' not in path:
                train_dataset = load_dataset(f'tasks/{task}.py')['train']
                assert len(train_dataset) == len(samples)
            else:
                train_dataset = None

            num_samples = len(samples)
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

            all_samples.extend(task_samples)

        print(len(all_samples))
        if "test" in path:
            # if len(tab_tasks) > 1:
            #     random.shuffle(new_samples)

            df = pd.DataFrame(all_samples)

            remain_keys = ['label', 'question', 'input', 'graph']
            sub_df = df[remain_keys]
            sub_df.to_parquet(f'{output_dir}/test.pq', engine='pyarrow', index=False)
            sub_df.to_parquet(f'{output_dir}/val.pq', engine='pyarrow', index=False)

            df_excluded = df.drop(columns=remain_keys)
            df_to_jsonl(df_excluded, f"{output_dir}/ori_test.jsonl")
            df_to_jsonl(df_excluded, f"{output_dir}/ori_val.jsonl")
        else:
            df = pd.DataFrame(all_samples)
            df.to_parquet(f'{output_dir}/train.pq', engine='pyarrow', index=False)

        print("done")