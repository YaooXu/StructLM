from copy import deepcopy
import random
import sys
import time
import zipfile

import numpy as np


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
from datasets import load_dataset, Dataset, DownloadMode
from torch.multiprocessing import Pool, Process, set_start_method
import torch.nn.functional as F

from dataset.data import TableConverter, GraphConverter
from utils.sentence_transformer import get_sentence_embeds

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
        # new_kg_tuples.append([f'Node: {tup[2]}', f'Inverse Relation: {tup[1]}', f'Node: {tup[0]}'])
    return new_kg_tuples


def obtain_samples(process_idx, idxes_to_process, cache_dir):
    if cache_graphs:
        llm = AutoModel.from_pretrained(
            model_path,
            device_map="auto",
        )
        llm.eval()

    new_samples = []
    tasks_to_n = defaultdict(int)
    t1 = time.time()
    for n, idx in enumerate(idxes_to_process):
        if (n + 1) % 500 == 0:
            t2 = time.time()
            print(f"{process_idx}: {n / len(idxes_to_process)}, {t2 - t1}")
            t1 = t2

        sample = all_samples[idx]

        try:
            if sample['key'] == 'table':
                graph = table_converter._text2graph(sample['table'], True)
            elif sample['key'] == 'schema_tuples':
                kg_graph = convert_kg_tups_bidir(sample["schema_tuples"])
                graph = graph_converter._kg_tupels2graph(kg_graph, True)
            elif sample['key'] == 'kg_tuples':
                kg_graph = convert_kg_tups_bidir(sample["kg_tuples"])
                graph = graph_converter._kg_tupels2graph(kg_graph, True)

            sample["graph"] = graph
            if graph:
                if cache_graphs:
                    graph = sample.pop("graph")

                    embedding_s = get_sentence_embeds(llm, tokenizer, torch.LongTensor(graph["x_s"]).to(llm.device), batch_size=2048)
                    graph["x_s"] = embedding_s.detach().cpu().float().numpy()

                    embedding_t = get_sentence_embeds(llm, tokenizer, torch.LongTensor(graph["x_t"]).to(llm.device), batch_size=2048)
                    graph["x_t"] = embedding_t.detach().cpu().float().numpy()

                    del embedding_s, embedding_t

                    zip_file_path = f"{cache_dir}/{process_idx}.zip"
                    with zipfile.ZipFile(zip_file_path, 'a', zipfile.ZIP_DEFLATED) as zipf:
                        graph_name = f"graph_{sample['idx']}.pkl"
                        sample["graph_path"] = (zip_file_path, graph_name)
                        
                        serialized_dict = pickle.dumps(graph)
                        zipf.writestr(graph_name, serialized_dict)
                else:
                    graph = sample.pop("graph")

                    zip_file_path = f"{cache_dir}/{process_idx}.zip"
                    graph_name = f"graph_{sample['idx']}.pkl"

                    sample["graph_path"] = (zip_file_path, graph_name)
                    new_samples.append(sample)
        except Exception as e:
            print(e)
            continue

    return new_samples

if __name__ == "__main__":

    n_process = 10 # DO NOT CHANGE!
    shuffle = False
    model_path = "sentence-transformers/all-roberta-large-v1"

    pretraining = False
    # output_dir = f"./data/hytrel/all-table-tasks-v2"
    output_dir = f"./data/hytrel/all-table-kg-schema-tasks"
    os.makedirs(output_dir, exist_ok=True)

    cache_graphs = True
    cache_dir = f"data/hytrel/all-table-kg-schema-tasks"

    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    table_converter, graph_converter = TableConverter(tokenizer), GraphConverter(tokenizer)

    for path in (
        "data/processed/custom_skginstruct.json",
        "data/processed/custom_test_skginstruct.json",
    ):
        all_samples = load_json(path=path)
        num_samples = len(all_samples)
        print(num_samples)

        if 'test' in path:
            cur_cache_dir = os.path.join(cache_dir, 'test')
        else:
            cur_cache_dir = os.path.join(cache_dir, 'train')

        with Pool(processes=n_process) as pool:
            num_samples_in_chunk = num_samples // n_process + 1
            jobs = []
            st = 0
            for i in range(n_process):
                ed = st + num_samples_in_chunk
                ed = min(ed, num_samples)
                jobs.append([i, list(range(st, ed)), cur_cache_dir])
                st = ed

            results = pool.starmap(obtain_samples, jobs)

        all_samples = []
        for samples in results:
            all_samples.extend(samples)

        print(len(all_samples))

        if "test" in path:
            df = pd.DataFrame(all_samples)

            remain_keys = ["label", "input", "graph_path"]
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
