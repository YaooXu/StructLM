import sys
sys.path.append('./')

import json
from datasets import load_dataset
import random
from tqdm import tqdm

from utils.tool import get_constructor
from utils.configure import Configure

from transformers import AutoTokenizer, AutoModel
from dataset.data import TableConverter, GraphConverter
model_path = "sentence-transformers/all-roberta-large-v1"
tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
table_converter, graph_converter = TableConverter(tokenizer), GraphConverter(tokenizer)
def convert_kg_tups_bidir(kg_tuples):
    new_kg_tuples = []
    for tup in kg_tuples:
        new_kg_tuples.append([f'Node: {tup[0]}', f'Relation: {tup[1]}', f'Node: {tup[2]}'])
        # new_kg_tuples.append([f'Node: {tup[2]}', f'Inverse Relation: {tup[1]}', f'Node: {tup[0]}'])
    return new_kg_tuples

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


train_processed_samples = []

table_tasks = ["wikitq", "hybridqa", "fetaqa", "tabmwp", "wikisql", "tab_fact", "totto", "kvret"]
schema_tasks = ["spider", "sparc"]
kg_tasks = ['compwebq', 'dart']


def construct_processed_samples(tasks, prompts_dict, is_train, output_path):
    processed_samples = []
    for task_name in tasks:
        error_cnt = 0

        print(task_name)
        raw_datasets_split = load_dataset(f"preprocess/tasks/{task_name}.py", trust_remote_code=True)
        datasets_split = get_constructor(f"preprocess.seq2seq_construction.{task_name}")(args).to_seq2seq(raw_datasets_split, './data/.cache')

        dataset = datasets_split[0] if is_train else datasets_split[-1]

        if task_name in table_tasks:
            key = 'table'
        elif task_name in schema_tasks:
            key = 'schema_tuples'
        elif task_name in kg_tasks:
            key = 'kg_tuples'

        for sample in tqdm(dataset):
            struct_data = sample[key]

            all_instructions = prompts_dict[task_name]['instruction']
            if isinstance(all_instructions, list):
                instruction = random.choice(all_instructions)
            elif isinstance(all_instructions, str):
                instruction = all_instructions

            input_format = prompts_dict[task_name]['input_format']

            text_in = sample['text_in']
            struct_in = sample['struct_in']
            seq_out = sample['seq_out']

            input_ = input_format.format(struct_in=struct_in, text_in=text_in)
            if '\n\n\n' in input_:
                input_ = input_.replace('\n\n\n', '[GRAPH_PAD]')
            else:
                input_ += '[GRAPH_PAD]'

            final_input = prompt.format(instruction=instruction, input=input_)
            label = seq_out

            if is_train:
                processed_sample = {
                    'idx': len(processed_samples),
                    'input': final_input,
                    'label': label,
                    key: struct_data,
                    'key': key
                }
            else:
                sample.update({
                    'idx': len(processed_samples),
                    'input': final_input,
                    'label': label, 
                    'key': key
                })
                processed_sample = sample
            processed_samples.append(processed_sample)

        #     try:
        #         if processed_sample['key'] == 'table':
        #             graph = table_converter._text2graph(processed_sample['table'], True)
        #         elif processed_sample['key'] == 'schema_tuples':
        #             kg_graph = convert_kg_tups_bidir(processed_sample["schema_tuples"])
        #             graph = graph_converter._kg_tupels2graph(kg_graph, True)
        #         elif processed_sample['key'] == 'kg_tuples':
        #             kg_graph = convert_kg_tups_bidir(processed_sample["kg_tuples"])
        #             graph = graph_converter._kg_tupels2graph(kg_graph, True)
        #     except Exception as e:
        #         table_converter._text2graph(processed_sample['table'], True)
        #         error_cnt += 1
        # print('#error: ', error_cnt, '\n')

        print(processed_samples[-1]['input'])

    print(len(processed_samples))
    with open(output_path, 'w') as f:
        json.dump(processed_samples, f)

all_tasks = table_tasks + schema_tasks + kg_tasks
construct_processed_samples(all_tasks, train_prompts_dict, True, 'data/processed/custom_skginstruct.json')
construct_processed_samples(all_tasks, test_prompts_dict, False, 'data/processed/custom_test_skginstruct.json')

