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
        uniskg_sample = uniskg_dataset[idx]
        
        if "test" in path:
            struct_data_key = 'table' if 'table' in sample else 'kg_tuples'

            question = sample['question'] if 'question' in sample else sample['statement']
            struct_data = sample[struct_data_key]
            sample['label'] = sample['seq_out']
            sample["input"] = sample["formatted_input"]
            # print(sample["formatted_input"])
            
            if shuffle:
                df = pd.DataFrame(sample['table']['rows'], columns=sample['table']['header'])
                # shuffle rows
                df_shuffled = df.sample(frac=1).reset_index(drop=True)
                
                # shuffle cols
                df_shuffled = df_shuffled.iloc[:, :].sample(frac=1, axis=1)
                new_rows = ["col : " + " | ".join(df_shuffled.columns)]

                for idx, row in df_shuffled.iterrows():
                    row_str = "row {} : ".format(idx + 1) + " | ".join(map(str, row.values))
                    new_rows.append(row_str)
                new_struct_in = " ".join(new_rows).lower()
                new_formatted_input = sample['formatted_input'].replace(sample['struct_in'], new_struct_in)

                if len(new_struct_in) != len(sample['struct_in']):
                    continue
                
                sample['input'] = new_formatted_input
                            
        else:
            train_data = train_dataset[idx]
            struct_data_key = 'table' if 'table' in train_data else 'kg_tuples'

            question = sample["input"].split('\n\n')[-1]
            assert question.lower().strip() == (train_data['question'] if 'question' in train_data else train_data['statement']).lower().strip()
            
            struct_data = train_data[struct_data_key]
            
            # table = re.findall(r"table:\n\n([\s\S]*)\n\n\n", sample["input"])[0]
            sys_prompt = 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n\n\n'
            sample["input"] = sys_prompt + sample["input"] + "\n\n### Response:\n"
            # print(sample['input'])

        sample["question"] = question
        sample['encoder_inputs'] = uniskg_sample

        new_samples.append(sample)
        
        if "test" in path:
            tasks_to_n[sample["description"]] += 1
        else:
            tasks_to_n[sample["task_name"]] += 1

    print(tasks_to_n)

    return new_samples


if __name__ == "__main__":

    output_dir = 'wikitq'
    output_dir = f'data/uniskg/{output_dir}'
    os.makedirs(output_dir, exist_ok=True)
    n_process = 32
    shuffle = False

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

    # converter = StructDataConverter(tokenizer)

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
                uniskg_dataset = torch.load('UniSKG/output/cache/wikitq_train.cache')
                train_dataset = load_dataset(f'tasks/{task}.py')['train']
                assert len(train_dataset) == len(samples)
            else:
                uniskg_dataset = torch.load('UniSKG/output/cache/wikitq_test.cache')
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

            remain_keys = ['label', 'question', 'input', 'encoder_inputs']
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
