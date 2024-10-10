

import sys
sys.path.append('./')

from datasets import load_dataset
from dataset.data import TableConverter, GraphConverter
from transformers import AutoTokenizer
import random
import numpy as np
import networkx as nx
import torch
from datasets import concatenate_datasets

def random_cell_index(num_rows, num_cols, ignore_first_col=False):
    row_index = random.randint(0, num_rows - 1)

    col_st_idx = 1 if ignore_first_col else 0
    col_index = random.randint(col_st_idx, num_cols - 1)

    return (row_index, col_index)


def sample_ignoring_element(lst, ignore_element):
    filtered_list = [elem for elem in lst if elem != ignore_element]

    sampled_element = random.choice(filtered_list)

    return sampled_element


def construct_table_pretraining_questions(table, k=10):
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
        function = random.choice(functions)
        qa_pairs = function()
        all_qa_pairs.extend(qa_pairs)

    return all_qa_pairs


def preprocess_table(samples, k=10):
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    table_converter = TableConverter(tokenizer)

    all_graphs, all_questions, all_labels = [], [], []
    for table in samples['table']:
        table['header'] = table.pop('headers')

        graph = table_converter._text2graph(table, return_dict=True)

        qa_pairs = construct_table_pretraining_questions(table, k=k)

        all_graphs.append(graph)
        all_questions.append([question for question, answer in qa_pairs])
        all_labels.append([answer for question, answer in qa_pairs])

    results = {
        "graph": all_graphs,
        "question": all_questions,
        "label": all_labels
    }

    return results


def compute_shortest_paths(i, j, edge_index):
    # Compute the shortest path between node i and node j
    # return all shortest paths and distance.
    G = nx.from_edgelist(edge_index.numpy().T)
    # check whether i or j is an isolated node or no path exists between i and j
    has_path = i in G and j in G and nx.has_path(G, i, j)
    if has_path:
        path_list = nx.all_shortest_paths(G, i, j)
        path_list = torch.tensor(list(path_list))
        return path_list, path_list.size(-1) - 1
    else:
        return [], 'inf'

def compute_common_neighbors(i, j, edge_index):
    i_neighbors = edge_index[1, edge_index[0] == i]
    j_neighbors = edge_index[1, edge_index[0] == j]
    cns = i_neighbors[torch.isin(i_neighbors, j_neighbors)]
    return cns.tolist()


def construct_graph_pretraining_questions(graph, k=10):
    edge_index, edge_attr, node_attr = graph['edge_index'], graph['edge_attr'], graph['node_attr']
    edge_index = torch.LongTensor(edge_index)

    sub_kg = {}
    for i, (u, v) in enumerate(edge_index.numpy().T):
        if u not in sub_kg:
            sub_kg[u] = {}
        
        rel = edge_attr[i]
        if rel not in sub_kg[u]:
            sub_kg[u][rel] = []
        
        sub_kg[u][rel].append(v)

    def construct_template1():
        template = "What's the {rel_j_node} of {node_i_text}?"

        i = random.choice(list(sub_kg.keys()))

        # 从 sub_kg[i] 中随机选择一个关系
        relation = random.choice(list(sub_kg[i].keys()))

        tails = sorted(sub_kg[i][relation])

        node_i_text=node_attr[i]
        question = template.format(rel_j_node=relation, node_i_text=node_i_text)

        answer = ', '.join([node_attr[j] for j in tails])

        return [(question, answer)]
    
    def construct_template2():
        template = "What's the relation between {node_i_text} and {node_j_text}?"

        i = random.choice(list(sub_kg.keys()))

        # 从 sub_kg[i] 中随机选择一个关系
        relation = random.choice(list(sub_kg[i].keys()))

        j = random.choice(sub_kg[i][relation])

        node_i_text, node_j_text = node_attr[i], node_attr[j]
        
        question = template.format(node_i_text=node_i_text, node_j_text=node_j_text)

        answer = relation

        return [(question, answer)]

    def construct_template3():
        template = "What's the shortest path distance between {node_i_text} and {node_j_text}?"

        i = random.choice(list(sub_kg.keys()))
        j = random.choice(list(sub_kg.keys()))

        node_i_text, node_j_text = node_attr[i], node_attr[j]

        question = template.format(node_i_text=node_i_text, node_j_text=node_j_text)

        path_list, spd = compute_shortest_paths(i, j, edge_index)

        if len(path_list):
            path_str_list = []
            for path in path_list:
                path = [node_attr[p] for p in path]
                path_str_list.append(", ".join(path))
            path_text = "; ".join(path_str_list)
            answer = f"Shortest paths: " + path_text + "."
        else:
            answer = f"There is no path between {node_i_text} and {node_i_text}."

        return [(question, answer)]
    

    def construct_template4():
        template = "What are the common neighbors between the node {node_i_text} and {node_j_text}?"

        i = random.choice(list(sub_kg.keys()))
        j = random.choice(list(sub_kg.keys()))
        cns = compute_common_neighbors(i, j, edge_index)

        node_i_text, node_j_text = node_attr[i], node_attr[j]
        question = template.format(node_i_text=node_i_text, node_j_text=node_j_text)

        if len(cns) == 0:
            answer = "There is no common neighbors between two nodes."
        else:
            cns_text = ', '.join([node_attr[x] for x in cns])
            answer = f"There are {len(cns)} common neighbors between two nodes, including {cns_text}."
        
        return [(question, answer)]

    functions = [construct_template1, construct_template2, construct_template3, construct_template4]
    all_qa_pairs = []

    for _ in range(k):
        function = random.choice(functions)
        qa_pairs = function()
        all_qa_pairs.extend(qa_pairs)

    return all_qa_pairs


def convert_graph_to_kg_tuples(graph):
    edge_index, edge_attr, node_attr = graph['edge_index'], graph['edge_attr'], graph['node_attr']

    kg_tuples = []
    for i, (u, v) in enumerate(np.array(edge_index).T):
        rel = edge_attr[i]
        head, tail = node_attr[u], node_attr[v]
        kg_tuples.append([head, rel, tail])
    return kg_tuples


def preprocess_graph(samples, k=20):
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    graph_converter = GraphConverter(tokenizer)

    all_graphs, all_questions, all_labels = [], [], []
    for ori_graph in samples['graph']:
        graph = graph_converter._kg_tupels2graph(convert_graph_to_kg_tuples(ori_graph), return_dict=True)

        qa_pairs = construct_graph_pretraining_questions(ori_graph, k=k)

        all_graphs.append(graph)
        all_questions.append([question for question, answer in qa_pairs])
        all_labels.append([answer for question, answer in qa_pairs])

    results = {
        "graph": all_graphs,
        "question": all_questions,
        "label": all_labels
    }

    return results


if __name__ == "__main__":
    model_path = "sentence-transformers/all-roberta-large-v1"
    num_proc = 1

    file_path = 'StructQformer/preprocess/table.pq'
    dataset = load_dataset("parquet", data_files=file_path)['train']
    dataset = dataset.select(range(10_000_000))
    processed_dataset1 = dataset.map(preprocess_table, batched=True, num_proc=num_proc, load_from_cache_file=False)
    print(len(processed_dataset1))

    processed_dataset1.to_parquet('data/hytrel/llm_pretraining_10M_tables/train.pq')

    print(processed_dataset1[0])

    # file_path = 'StructQformer/preprocess/graph_1.2M.pq'
    # dataset = load_dataset("parquet", data_files=file_path)['train']
    # processed_dataset2 = dataset.map(preprocess_graph, batched=True, num_proc=num_proc, load_from_cache_file=False)
    # print(len(processed_dataset2))
    
    # # repeat n times
    # N = 2
    # merged_dataset = concatenate_datasets([processed_dataset1] +  [processed_dataset2] * n)

    # # # 查看合并后的数据集
    # # print(merged_dataset)
    # merged_dataset.to_parquet('data/hytrel/pretraining_10M_tables_1M_graph/train.pq')
