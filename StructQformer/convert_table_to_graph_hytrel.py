from copy import deepcopy
import random
import sys
import time
import zipfile

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


def convert_totto_table_format(table):
    # 计算表格的列数
    # num_columns = max([sum(cell["column_span"] for cell in row) for row in table])
    num_columns = sum(cell["column_span"] for cell in table[0])
    
    try:
        # 初始化一个空的表格结构，用于放置最终的 Markdown 表格数据
        table_structure = [[""] * num_columns for _ in range(len(table))]
        
        for row_idx, row in enumerate(table):
            col_idx = 0
            for cell in row:
                while table_structure[row_idx][col_idx] != "":
                    col_idx += 1
                
                # 将单元格的值填入表格结构中
                table_structure[row_idx][col_idx] = cell["value"]
                
                # 如果 column_span 或 row_span 大于 1，则填充跨越的单元格
                for i in range(cell["row_span"]):
                    if row_idx + i >= len(table_structure):
                        break
                    
                    for j in range(cell["column_span"]):
                        if i == 0 and j == 0:
                            continue
                        if col_idx + j >= num_columns:
                            break
                        
                        table_structure[row_idx + i][col_idx + j] = cell["value"]

        return {
            'header': table_structure[0],
            'rows': table_structure[1:]
        }
    except:
        return {
            'header': None,
            'rows': None
        }

class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_index = torch.LongTensor(edge_index) if edge_index else None  # [2, N]
        self.x_s = torch.Tensor(x_s) if edge_index else None
        self.x_t = torch.Tensor(x_t) if edge_index else None

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
            if 'question' in sample:
                question = sample['question']
            elif 'statement' in sample:
                question = sample['statement']
            elif 'hypothesis' in sample:
                question = sample['hypothesis']
            else:
                # totto
                question = 'What the table snippet describes?'
                sample["formatted_input"] = sample["formatted_input"].replace('### Response:\n', '').strip()
                sample["formatted_input"] += f"\n\n\nquestion:\n\n{question}\n\n### Response:\n"

                sample["table"] = convert_totto_table_format(sample["table"])

            sample['question'] = question
            struct_data = sample["table"]
            sample["label"] = sample["seq_out"]
            sample["input"] = sample["formatted_input"]

            idx = sample["input"].rfind('\n\n\n')
            # ensure not the \n\n\n before Instruction
            assert idx >= 200
            sample["input"] = sample["input"][:idx] + '[GRAPH_PAD]' + sample["input"][idx+3:]

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

            if "question" in train_data:
                ori_question = train_data["question"]
            elif "statement" in train_data:
                ori_question = train_data["statement"]
            else:
                # no question in dataset like totto
                ori_question = None

            if ori_question:
                question = sample["input"].split("\n\n")[-1]
                assert question.lower().strip() == ori_question.lower().strip()
            else:
                # totto
                question = 'What the table snippet describes?'
                sample["input"] += f'\n\n\nquestion:\n\n{question}'
                train_data["table"] = convert_totto_table_format(train_data["table"])

            sample['question'] = question

            struct_data = train_data["table"]

            # table = re.findall(r"table:\n\n([\s\S]*)\n\n\n", sample["input"])[0]
            sys_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n\n\n"

            sample["input"] = sample["input"].replace('\n\n\n', '[GRAPH_PAD]')
            sample["input"] = sys_prompt + sample["input"] + "\n\n### Response:\n"

        assert '[GRAPH_PAD]' in sample['input']
        # print(sample['input'])
        # print(question)
        # print(sample['label'])
        try:
            graph = converter._text2graph(struct_data, True)
            sample["graph"] = graph
            if graph:
                if not pretraining:

                    new_samples.append(sample)

                    if "test" in path:
                        tasks_to_n[sample["description"]] += 1
                    else:
                        tasks_to_n[sample["task_name"]] += 1
                else:
                    # to save memory
                    sample.pop('graph')

                    # construct self-supervised data
                    qa_pairs = construct_pretraining_questions(struct_data)
                    for qa_pair in qa_pairs:
                        new_sample = deepcopy(sample)

                        new_sample["question"] = qa_pair[0]
                        new_sample["label"] = new_sample["seq_out"] = qa_pair[1]

                        # # replace original question with new question
                        # new_sample["input"] = new_sample["input"].replace(sample['question'], new_sample["question"])

                        # only remain question in pretraining
                        new_sample["input"] = f"Question: {new_sample['question']}\n\n### Response:\n"

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


def construct_pretraining_questions(table, k=20):
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
    model_path = "sentence-transformers/all-roberta-large-v1"

    pretraining = False
    output_dir = f"./data/hytrel/all-table-tasks"

    # pretraining = True
    # output_dir = f"./data/hytrel/pretraining"

    cache_graphs = False
    cache_dir = f"./data/hytrel/cache2"
    if cache_graphs:
        llm = AutoModel.from_pretrained(
            model_path,
            device_map="auto",
        )
        llm.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    converter = TableConverter(tokenizer)
    for path, tab_tasks in (
        (
            "data/processed/skginstruct_skgonly.json",
            ["fetaqa", "hybridqa", "wikitq", "tabmwp", "totto", "wikisql", "tab_fact"],
        ),
        (
            "data/processed/skginstruct_test_file_mistral.json",
            ["task: wiki table question", "task: fetaqa", "task: hybridqa", "task: totto", "task: wikisql", "task: tabfact"]
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
                train_dataset = load_dataset(f"tasks/{task}.py", trust_remote_code=True)["train"]
                assert len(train_dataset) == len(samples)
            else:
                train_dataset = None

            num_samples = len(samples)
            print(task, num_samples)

            # obtain_samples(0, [1,])

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

            if cache_graphs:
                os.makedirs(f"{cache_dir}/{task}", exist_ok=True)

                with torch.no_grad():
                    for sample in tqdm(task_samples):
                        # replace graph with graph path
                        graph = sample.pop("graph")

                        embedding_s = get_sentence_embeds(llm, tokenizer, graph["x_s"])
                        graph["x_s"] = embedding_s.detach().cpu().float().numpy()

                        embedding_t = get_sentence_embeds(llm, tokenizer, graph["x_t"])
                        graph["x_t"] = embedding_t.detach().cpu().float().numpy()

                        del embedding_s, embedding_t

                        zip_file_path = f"{cache_dir}/{task}/{sample['idx'] // 1000}.zip"
                        with zipfile.ZipFile(zip_file_path, 'a', zipfile.ZIP_DEFLATED) as zipf:
                            graph_name = f"graph_{sample['idx']}.pkl"
                            sample["graph_path"] = (zip_file_path, graph_name)
                            
                            serialized_dict = pickle.dumps(graph)
                            zipf.writestr(graph_name, serialized_dict)
            else:
                for sample in tqdm(task_samples):
                    graph = sample.pop("graph") if "graph" in sample else None

                    zip_file_path = f"{cache_dir}/{task}/{sample['idx'] // 1000}.zip"
                    graph_name = f"graph_{sample['idx']}.pkl"

                    sample["graph_path"] = (zip_file_path, graph_name)
                
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
