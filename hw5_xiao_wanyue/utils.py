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


if __name__ == "__main__":
    pass
