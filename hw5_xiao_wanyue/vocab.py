import os
import json
import re
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter as counter
from typing import Dict, Iterable, Union, Generator


def load_wapo(wapo_jl_path: Union[str, os.PathLike]) -> Generator[Dict, None, None]:
    print("load_wapo")
    counter = 0

    with open(wapo_jl_path, 'r', encoding='UTF-8') as file:
        for line in file:
            if(line):
                dictData = {}
                conv = json.loads(line)

                dictData["title"] = conv.get("title")
                dictData["author"] = conv.get("author")
                dictData["content"] = conv.get("content_str")
                counter += 1
                yield dictData


def process(txt):
    """
    Tokenize an input string. Something more sophisticated may help . . .
    """
    txt = re.sub(r"[^a-zA-Z0-9\-\'\"\s]", "", str(txt))
    return txt


def build_vocabulary(docs: Iterable):
    """
    Inputs: arrays representing the training, validation and test data
    Outputs: vocabulary (Tokenized text as in-place modification of input arrays or returned as new arrays)
    """
    all_tokens = []
    docs = list(docs)
    for i, instance in enumerate(docs):
        title = instance["title"] if instance["title"] is not None else ""
        content = instance["content"] if instance["content"] is not None else ""
        author = instance["author"] if instance["author"] is not None else ""
        tokens = word_tokenize(content.lower()) + title.lower().split(" ") + author.lower().split(" ") + [author.lower()]
        for tok in tokens:
            if len(tok) > 2:
                all_tokens.append(process(tok))

        if i % 10000 == 0:
            print(i, ' has been tokenized')

    sw = stopwords.words('english')
    vocab_counter = counter(all_tokens)
    vocab_counter = dict((key, value) for key, value in vocab_counter.items() if key not in sw)
    return vocab_counter


def trigger():
    data_dir = Path(__file__).parent.joinpath("pa5_data")
    wapo_path = data_dir.joinpath("subset_wapo_50k_sbert_ft_filtered.jl")
    vocabulary = build_vocabulary(load_wapo(wapo_path))

    with open("./pa5_data/vocab_list.txt", 'w') as output:
        for k in vocabulary.keys():
            output.write("{}: {} \n".format(str(k), int(vocabulary[k])))
        output.close()

    with open("./pa5_data/vocab_json.json", "w") as fp:
        json.dump(vocabulary, fp)

if __name__ == "__main__":
    pass