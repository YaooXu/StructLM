from copy import deepcopy
import json
import re
import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm

from utils import load_json

SPARQLPATH = "http://210.75.240.139:18890/sparql"
sparql = SPARQLWrapper(SPARQLPATH)
sparql.setReturnFormat(JSON)
ns_url_prefix = "http://rdf.freebase.com/ns/"


samples = load_json("cwq_samples.json")


def remove_ns(id):
    if id.startswith("ns:"):
        return id[3:]
    else:
        return id


def convert_id_to_name(entity_id):
    entity_id = remove_ns(entity_id)

    sparql_id = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?tailEntity
    WHERE {{
        {{
            ?entity ns:type.object.name ?tailEntity .
            FILTER(?entity = ns:%s)
        }}
        UNION
        {{
            ?entity ns:common.topic.alias ?tailEntity .
            FILTER(?entity = ns:%s)
        }}
    }}
    """
    sparql_query = sparql_id % (entity_id, entity_id)

    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    if len(results["results"]["bindings"]) == 0:
        return entity_id
    else:
        return results["results"]["bindings"][0]["tailEntity"]["value"]


def convert_id_to_name_in_triples(triples, return_map=False):
    id_to_label = {}
    for triple in triples:
        for i in [0, -1]:
            ent_id = triple[i]
            if ent_id[:2] in ["m.", "g."]:
                if ent_id not in id_to_label:
                    id_to_label[ent_id] = convert_id_to_name(ent_id)
                triple[i] = id_to_label[ent_id]

    if return_map:
        return triples, id_to_label
    else:
        return triples


def get_all_bound_triples(
    data_path="data/ComplexWebQuestions/ComplexWebQuestions_dev.json",
):
    with open(data_path, "r") as f:
        samples = json.load(f)
    print(len(samples))

    samples_w_bt = []
    for sample in tqdm(samples):
        # topic_mids = [mid for mid, label in sample["topic_entity"].items()]

        # if len(topic_mids) == 0:
        #     print(sample)

        if "sparql" in sample:
            sparql_query = sample["sparql"]
        else:
            sparql_query = sample["Parses"][0]["Sparql"]
        lines = sparql_query.split("\n")

        # TODO: some entities only appear in FILTER
        triples = re.findall(
            r"(\?\w+|ns:[\w.:]+)\s+(ns:[\w.:]+)\s+(\?\w+|ns:[\w.:]+)", sparql_query
        )
        triples = [list(triple) for triple in triples]

        variables = re.findall(r"(\?\w+)", sparql_query)
        variables = sorted(list(set(variables)))

        for i, line in enumerate(lines):
            if line.startswith("SELECT DISTINCT"):
                parts = line.split()
                parts = parts[:2] + variables
                lines[i] = " ".join(parts)
            elif "LIMIT 1" in line:
                lines[i] = ""

        sparql_query = "\n".join(lines)
        print(sparql_query)

        sparql.setQuery(sparql_query)
        results = sparql.query().convert()

        try:
            all_bound_mid_triples = []
            all_bound_triples = []
            all_topic_node_to_path = []
            for binding in results["results"]["bindings"]:
                ans = binding["x"]["value"].replace(ns_url_prefix, "")
                bound_triples = []
                for triple in triples:
                    bound_triple = triple.copy()

                    if triple[0].startswith("?") and triple[0][1:] in binding:
                        bound_triple[0] = binding[triple[0][1:]]["value"].replace(
                            ns_url_prefix, "ns:"
                        )

                    if triple[-1].startswith("?") and triple[-1][1:] in binding:
                        bound_triple[-1] = binding[triple[-1][1:]]["value"].replace(
                            ns_url_prefix, "ns:"
                        )

                    # remove all ns prefix:
                    for i in range(3):
                        bound_triple[i] = remove_ns(id=bound_triple[i])

                    if bound_triple[0].startswith("?") or bound_triple[-1].startswith("?"):
                        # ignore unbound triple without bound var
                        continue

                    bound_triples.append(bound_triple)

                all_bound_mid_triples.append(bound_triples)
                all_bound_triples.append(convert_id_to_name_in_triples(deepcopy(bound_triples)))

            #     topic_node_to_path = bfs_shortest_paths(bound_triples, ans, topic_mids)

            #     if len(topic_node_to_path):
            #         all_topic_node_to_path.append(topic_node_to_path)
            #         all_bound_triples.append(bound_triples)

            # if len(all_topic_node_to_path) == 0:
            #     print(all_topic_node_to_path)

            # crucial_edges = get_crucial_edges(
            #     all_topic_node_to_path,
            #     n_considered_source_nodes=n_considered_source_nodes,
            #     n_edge_to_drop=n_edge_to_drop,
            # )
        except Exception as e:
            print(e)

        sample["all_bound_mid_triples"] = all_bound_mid_triples
        sample["all_bound_triples"] = all_bound_triples
        
        samples_w_bt.append(sample)

    return samples_w_bt


if __name__ == "__main__":
    samples = get_all_bound_triples()
    print(len(samples))
    with open("data/ComplexWebQuestions/ComplexWebQuestions_dev.bound_triples.jsonl", "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
