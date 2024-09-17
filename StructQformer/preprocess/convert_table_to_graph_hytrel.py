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
from transformers import AutoTokenizer, AutoModel
from multiprocessing import Pool
from datasets import load_dataset, Dataset
from torch.multiprocessing import Pool, Process, set_start_method
import torch.nn.functional as F

from StructQformer.utils.data import TableConverter, GraphConverter
from StructQformer.utils.sentence_transformer import get_sentence_embeds

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

def convert_kg_tups_bidir(kg_tuples):
    new_kg_tuples = []
    for tup in kg_tuples:
        new_kg_tuples.append([f'Node: {tup[0]}', f'Relation: {tup[1]}', f'Node: {tup[2]}'])
        new_kg_tuples.append([f'Node: {tup[2]}', f'Inverse Relation: {tup[1]}', f'Node: {tup[0]}'])
    return new_kg_tuples


def obtain_samples(process_idx, idxes_to_process, task_name):
    new_samples = []
    tasks_to_n = defaultdict(int)
    t1 = time.time()
    for n, idx in enumerate(idxes_to_process):
        if (n + 1) % 500 == 0:
            t2 = time.time()
            print(f"{process_idx}: {n / len(idxes_to_process)}, {t2 - t1}")
            t1 = t2

        sample = samples[idx]
        sample["idx"] = idx
        is_table, is_graph = None, None

        if "test" in path:
            if 'question' in sample:
                question = sample['question']
            elif 'statement' in sample:
                question = sample['statement']
            elif 'hypothesis' in sample:
                question = sample['hypothesis']
            else:
                if 'totto' in task_name:
                    question = 'What the table snippet describes?'
                    sample["table"] = convert_totto_table_format(sample["table"])
                elif 'dart' in task_name:
                    question = 'What the triples describes?'

                sample["formatted_input"] = sample["formatted_input"].replace('### Response:\n', '').strip()
                sample["formatted_input"] += f"\n\n\nquestion:\n\n{question}\n\n### Response:\n"

            if 'table' in sample:
                struct_data = sample["table"]
                is_table = True
            elif 'kg_tuples' in sample:
                struct_data = convert_kg_tups_bidir(sample["kg_tuples"])
                is_graph = True

            sample['question'] = question
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
                if task_name == 'totto':
                    question = 'What the table snippet describes?'
                    train_data["table"] = convert_totto_table_format(train_data["table"])
                elif task_name == 'dart':
                    question = 'What the triples describes?'

                sample["input"] += f'\n\n\nquestion:\n\n{question}'

            sample['question'] = question

            if 'table' in train_data:
                struct_data = train_data["table"]
                is_table = True
            elif 'kg_tuples' in train_data:
                struct_data = convert_kg_tups_bidir(train_data["kg_tuples"])
                is_graph = True

            # table = re.findall(r"table:\n\n([\s\S]*)\n\n\n", sample["input"])[0]
            sys_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n\n\n"

            sample["input"] = sample["input"].replace('\n\n\n', '[GRAPH_PAD]')
            sample["input"] = sys_prompt + sample["input"] + "\n\n### Response:\n"

        assert '[GRAPH_PAD]' in sample['input']

        try:
            if is_table:
                graph = table_converter._text2graph(struct_data, True)
            elif is_graph:
                graph = graph_converter._kg_tupels2graph(struct_data, True)

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

if __name__ == "__main__":

    n_process = 16
    shuffle = False
    model_path = "sentence-transformers/all-roberta-large-v1"

    pretraining = False
    # output_dir = f"./data/hytrel/all-table-tasks-v2"
    output_dir = f"./data/hytrel/all-table-kg-tasks-v2"
    os.makedirs(output_dir, exist_ok=True)

    # pretraining = True
    # output_dir = f"./data/hytrel/pretraining"

    cache_graphs = False
    cache_dir = f"data/hytrel/all-table-kg-tasks/cache"
    if cache_graphs:
        llm = AutoModel.from_pretrained(
            model_path,
            device_map="auto",
        )
        llm.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    table_converter, graph_converter = TableConverter(tokenizer), GraphConverter(tokenizer)

    for path, tab_tasks in (
        (
            "data/processed/skginstruct_skgonly.json",
            # ["fetaqa", "hybridqa", "wikitq", "tabmwp", "totto", "wikisql", "tab_fact"],
            ["wikitq", "hybridqa", "fetaqa", "tabmwp", "totto", "wikisql", "tab_fact", "compwebq", "dart"],
        ),
        (
            "data/processed/skginstruct_test_file_mistral.json",
            # ["task: fetaqa", "task: hybridqa", "task: wiki table question", "task: tabmwp", "task: totto", "task: wikisql", "task: tabfact"]
            ["task: fetaqa", "task: hybridqa", "task: wiki table question", "task: tabmwp", "task: totto", "task: wikisql", "task: tabfact", "task: compwebq", 'task: dart']
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
        for task_name in tab_tasks:
            samples = tasks_to_samples[task_name]

            if "test" not in path:
                train_dataset = load_dataset(f"tasks/{task_name}.py", trust_remote_code=True)["train"]
                assert len(train_dataset) == len(samples)
            else:
                train_dataset = None

            num_samples = len(samples)
            print(task_name, num_samples)

            with Pool(processes=n_process) as pool:
                num_samples_in_chunk = num_samples // n_process + 1
                jobs = []
                st = 0
                for i in range(n_process):
                    ed = st + num_samples_in_chunk
                    ed = min(ed, num_samples)
                    jobs.append([i, list(range(st, ed)), task_name])
                    st = ed

                results = pool.starmap(obtain_samples, jobs)

            task_samples = []
            for samples in results:
                task_samples.extend(samples)
            print(len(task_samples))

            print(task_samples[0]['input'])

            if cache_graphs:
                os.makedirs(f"{cache_dir}/{task_name}", exist_ok=True)

                with torch.no_grad():
                    for sample in tqdm(task_samples):
                        # replace graph with graph path
                        graph = sample.pop("graph")

                        embedding_s = get_sentence_embeds(llm, tokenizer, graph["x_s"])
                        graph["x_s"] = embedding_s.detach().cpu().float().numpy()

                        embedding_t = get_sentence_embeds(llm, tokenizer, graph["x_t"])
                        graph["x_t"] = embedding_t.detach().cpu().float().numpy()

                        del embedding_s, embedding_t

                        zip_file_path = f"{cache_dir}/{task_name}/{sample['idx'] // 1000}.zip"
                        with zipfile.ZipFile(zip_file_path, 'a', zipfile.ZIP_DEFLATED) as zipf:
                            graph_name = f"graph_{sample['idx']}.pkl"
                            sample["graph_path"] = (zip_file_path, graph_name)
                            
                            serialized_dict = pickle.dumps(graph)
                            zipf.writestr(graph_name, serialized_dict)
            else:
                pass
                # for sample in tqdm(task_samples):
                #     graph = sample.pop("graph") if "graph" in sample else None

                #     zip_file_path = f"{cache_dir}/{task_name}/{sample['idx'] // 1000}.zip"
                #     graph_name = f"graph_{sample['idx']}.pkl"

                #     sample["graph_path"] = (zip_file_path, graph_name)
                
            all_samples.extend(task_samples)

        print(len(all_samples))

        if "test" in path:
            df = pd.DataFrame(all_samples)

            remain_keys = ["label", "question", "input", "graph"]
            sub_df = df[remain_keys]

            dataset = Dataset.from_pandas(sub_df)
            dataset.to_parquet(f"{output_dir}/test.pq")
            dataset.to_parquet(f"{output_dir}/val.pq")

            df_excluded = df.drop(columns=remain_keys)
            df_to_jsonl(df_excluded, f"{output_dir}/ori_test.jsonl")
            df_to_jsonl(df_excluded, f"{output_dir}/ori_val.jsonl")
        else:
            df = pd.DataFrame(all_samples)
            dataset = Dataset.from_pandas(df)
            dataset.to_parquet(f"{output_dir}/train.pq")

            # from datasets import load_dataset
            # load_dataset("parquet", data_files=f"{output_dir}/train.pq")['train']

        print("done")
