import sys

sys.path.append('./')

from preprocess.construct_pretrain_data import construct_table_pretraining_questions
import json
from datasets import load_dataset
import random
from tqdm import tqdm

from utils.tool import get_constructor
from utils.configure import Configure

from transformers import AutoTokenizer, AutoModel

from datasets import load_dataset
from dataset.data import TableConverter, GraphConverter
from transformers import AutoTokenizer
import random
import numpy as np
import networkx as nx
import torch
from datasets import concatenate_datasets


def preprocess_table(samples, k=10):
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    table_converter = TableConverter(tokenizer)

    all_tables, all_questions, all_labels = [], [], []

    dataset = [] 
    for table in samples['table']:

        qa_pairs = construct_table_pretraining_questions(table, k=k)
        qa_pairs = [(question, answer) for question, answer in qa_pairs if answer != '']

        dataset.append({
            "table": table,
            "question": [question for question, answer in qa_pairs],
            "answer_text": [answer for question, answer in qa_pairs],
        })

    processed_samples = construct_processed_samples(dataset, train_prompts_dict)

    results = {key: [d[key] for d in processed_samples] for key in processed_samples[0]}

    return results

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto")
def truncate_to_max_tokens(text, max_tokens=2048):
    # 使用tokenizer进行token化，确保返回的tokens不会超过 max_tokens
    tokens = tokenizer(text, truncation=True, max_length=max_tokens, return_tensors='pt')

    # 将这些tokens转回原始文本
    truncated_text = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
    return truncated_text

prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. "\
        "Write a response that appropriately completes the request.\n\n"\
        "### Instruction:\n\n\n{instruction}\n\n{input}\n\n### Response:\n"

args = Configure.Get('construct_sft_data.cfg')

train_prompt_filepath = 'prompts/instuning_format_spec.json'
with open(train_prompt_filepath, 'r') as f:
    train_prompts_dict = json.load(f)

test_prompt_filepath = 'prompts/instuning_format_spec_eval.json'
with open(test_prompt_filepath, 'r') as f:
    test_prompts_dict = json.load(f)


def construct_processed_samples(dataset, prompts_dict, is_train=True):
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
    table_converter = TableConverter(tokenizer)

    args = Configure.Get('construct_sft_data.cfg')

    processed_samples = []

    dataset = get_constructor(f"preprocess.seq2seq_construction.pretraining")(args).to_seq2seq(dataset)
    key = 'table'
    task_name = 'wikitq'

    for sample in dataset:
        struct_data = sample[key]

        all_instructions = prompts_dict[task_name]['instruction']
        if isinstance(all_instructions, list):
            instruction = random.choice(all_instructions)
        elif isinstance(all_instructions, str):
            instruction = all_instructions

        input_format = prompts_dict[task_name]['input_format']

        text_in_list = sample['text_in']
        struct_in = truncate_to_max_tokens(sample['struct_in'])
        seq_out_list = sample['seq_out']

        labels = []
        inputs = []
        for text_in, seq_out in zip(text_in_list, seq_out_list):
            input_ = input_format.format(struct_in=struct_in, text_in=text_in)
            if '\n\n\n' in input_:
                input_ = input_.replace('\n\n\n', '[GRAPH_PAD]')
            else:
                input_ += '[GRAPH_PAD]'
            
            assert '[GRAPH_PAD]' in input_

            final_input = prompt.format(instruction=instruction, input=input_)

            labels.append(seq_out)
            inputs.append(final_input)

        task_sample = {
            'idx': len(processed_samples),
            'input': inputs,
            'label': labels,
            'question': sample['question'],
            key: struct_data,
            'key': key,
            'task': task_name,
            'task_id': len(processed_samples),
        }

        graph = table_converter._text2graph(sample[key], True)
        
        if graph:
            task_sample['graph'] = graph
            processed_samples.append(task_sample)

    return processed_samples

if __name__ == "__main__":
    model_path = "sentence-transformers/all-roberta-large-v1"
    num_proc = 64

    file_path = '/cpfs/29f69eb5e2e60f26/code/pretrain/xuyao/TaBERT/data/pretrain/data.pq'
    dataset = load_dataset("parquet", data_files=file_path)['train']
    processed_dataset1 = dataset.map(preprocess_table, batched=True, num_proc=num_proc, load_from_cache_file=False)
    filter_dataset1 = processed_dataset1.filter(lambda sample: len(sample['question']) > 0, num_proc=num_proc)
    print(len(filter_dataset1))
    filter_dataset1.to_parquet('data/llm_based_pretraining/train.pq')
