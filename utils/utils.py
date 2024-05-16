import json

instruct = (
    "[INST] <<SYS>>\nYou are an AI assistant that specializes in analyzing and reasoning over structured information. "
    "You will be given a task, optionally with some structured knowledge input. Your answer must strictly adhere to the output format, if specified.\n<</SYS>>\n\n"
)

question_pattern = "Answer the following question with the help of the given list of the knowledge graph triples. knowledge graph triples:\n{kg_tuples}\nquestion:{question}"


def convert_kg_tuples_to_str(all_triples):
    kg_tuples = []
    for triple in all_triples:
        kg_tuples.append(" ".join(triple))
    kg_tuples = " | ".join(kg_tuples)
    return kg_tuples


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_jsonl(path):
    datas = []
    with open(path, "r") as f:
        for line in f:
            datas.append(json.loads(line))
    return datas


def write_jsonl(path, samples):
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
