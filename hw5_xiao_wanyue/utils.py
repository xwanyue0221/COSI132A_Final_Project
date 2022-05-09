from typing import Dict, Union, Generator
import os
import json


def load_clean_wapo_with_embedding(wapo_jl_path: Union[str, os.PathLike]) -> Generator[Dict, None, None]:
    """
    load wapo docs as a generator
    :param wapo_jl_path:
    :return: yields each document as a dict
    """
    with open(wapo_jl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            yield json.loads(line)


def load_topic_queries(query_json_file: str) -> Dict[str, Dict[str, str]]:
    with open(query_json_file, "r") as f:
        query_lst = json.load(f)["pa5_queries"]
    return {k["topic"]: k for k in query_lst}


def concate_topic_embedding(wapo_path: str, topic_emb_path: str, new_wapo_path: str) -> None:
    wp = open(wapo_path, "r")
    tp = open(topic_emb_path, "r")
    topic_embs = json.load(tp)
    with open(new_wapo_path, "w") as f:
        for i, line in enumerate(wp):
            line = json.loads(line)
            line["topic_vector"] = topic_embs[f"{i}"]
            json.dump(line, f)
            f.write('\n')
    wp.close()
    tp.close()
    return


if __name__ == "__main__":
    wapo_path = "pa5_data/subset_wapo_50k_sbert_ft_filtered.jl"
    topic_emb_path = "pa5_data/doc_vectors.json"
    new_wapo_path = "pa5_data/subset_wapo_50k_sbert_ft_lda_filtered.jl"
    concate_topic_embedding(wapo_path, topic_emb_path, new_wapo_path)