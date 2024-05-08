import itertools
import json
import numpy as np
from utils import load_json, load_jsonl, question_pattern, instruct, convert_kg_tuples_to_str


if __name__ == "__main__":
    data_filepath = "data/ComplexWebQuestions/ComplexWebQuestions_dev.syn_bound_triples.jsonl"

    samples = load_jsonl(data_filepath)

    examples = []
    for sample in samples[:2]:
        instruct_sample = {}

        question = sample["question"]

        instruct_sample["answers"] = [answer["answer"] for answer in sample["answers"]]
        answers = ", ".join(instruct_sample["answers"])

        all_triples = []
        for bound_triples in sample["all_bound_triples"]:
            all_triples.extend(bound_triples)

        kg_tuples = convert_kg_tuples_to_str(all_triples)
        example = (
            question_pattern.format(kg_tuples=kg_tuples, question=question)
            + f"\nanswers: {answers}"
        )
        examples.append(example)

    examples = "Here are some examples:\n" + "\n".join(examples) + "\n"

    instruct_samples = []
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

        all_triples = list(itertools.chain(*sample["all_bound_triples"]))

        instruct_sample["kg_tuples"] = all_triples

        kg_tuples = convert_kg_tuples_to_str(instruct_sample["kg_tuples"])

        instruct_sample["formatted_input"] = (
            instruct
            + examples
            + question_pattern.format(kg_tuples=kg_tuples, question=sample["question"])
            + f"\nanswers: "
        )

        instruct_samples.append(instruct_sample)

    with open("data/processed/syn_cwq_skginstruct_w_ICL.json", "w") as f:
        json.dump(instruct_samples, f)
