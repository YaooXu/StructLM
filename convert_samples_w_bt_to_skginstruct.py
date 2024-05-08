import itertools
import json
import numpy as np
from utils import load_json, load_jsonl, convert_kg_tuples_to_str, instruct, question_pattern

# instruct = (
#     "[INST] <<SYS>>\nYou are an AI assistant that specializes in analyzing and reasoning over structured information. "
#     "You will be given a task, optionally with some structured knowledge input. Your answer must strictly adhere to the output format, if specified.\n<</SYS>>\n\n"
#     "Answer the following question with the help of the given list of the knowledge graph triples. knowledge graph triples:\n\n{kg_tuples}\n\n\nquestion:\n\n{question} [/INST]"
# )

if __name__ == "__main__":
    data_filepath = "data/ComplexWebQuestions/ComplexWebQuestions_dev.syn_bound_triples.jsonl"

    samples = load_jsonl(data_filepath)
    instruct_samples = []
    n = []
    for sample in samples:
        instruct_sample = {
            "description": "task: compwebq",
            "section": "test",
            "arg_path": "META_TUNING/compwebq.cfg",
        }
        instruct_sample["id"] = sample["ID"]
        instruct_sample["question"] = sample["question"]
        instruct_sample["answers"] = [answer["answer"] for answer in sample["answers"]]
        instruct_sample["seq_out"] = ", ".join(instruct_sample["answers"])

        sample["all_bound_triples"] = sample["all_bound_triples"][:10]

        n.append(len(sample["all_bound_triples"]))

        all_triples = list(itertools.chain(*sample["all_bound_triples"]))

        instruct_sample["kg_tuples"] = all_triples

        kg_tuples = convert_kg_tuples_to_str(instruct_sample["kg_tuples"])

        instruct_sample["formatted_input"] = instruct + question_pattern.format(
            kg_tuples=kg_tuples, question=sample["question"]
        ) + ' [/INST]'

        instruct_samples.append(instruct_sample)

    with open("data/processed/syn_cwq_skginstruct.json", "w") as f:
        json.dump(instruct_samples, f)
    print('finish')