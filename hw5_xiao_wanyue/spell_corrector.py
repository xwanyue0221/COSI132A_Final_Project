import time
import logging
import json
from typing import List
import os
from vocab import trigger

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


class SpellCorrector():
    def __init__(self):
        if os.path.exists('./pa5_data/vocab_json.json'):
            print(" The file exists")
            with open('./pa5_data/vocab_json.json') as json_file:
                content = json.load(json_file)
            self.vocabulary = dict((key, value) for key, value in content.items() if 'www' not in key)
        else:
            trigger()
            with open('./pa5_data/vocab_json.json') as json_file:
                content = json.load(json_file)
            self.vocabulary = dict((key, value) for key, value in content.items() if 'www' not in key)


    def correct(self, word: str):
        word = word.lower().strip()

        if self.known(word):
            return word
        elif self.known(self.edits1(word)):
            candidate = self.known(self.edits1(word))
        elif self.known_edits2(word):
            candidate = self.known_edits2(word)
        else:
            return word

        output_dic = {key: self.vocabulary[key] for key in candidate}
        max_key = max(output_dic, key=output_dic.get)
        return max_key

    def edits1(self, word: str) -> List[str]:
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        alphabet = list(alphabet.lower().strip())

        edits = []
        each = word.lower()
        total = len(each)
        # print(each, total)

        for i in range(total):
            # deleting one char
            if len(each[:i] + each[i+1:]) > 2:
                edits.append(each[:i] + each[i+1:])
            for char in alphabet:
                # substituting one char
                edits.append(each[:i] + char + each[i+1:])
        # swapping chars order
        for i in range(total-1):
            edits.append(each[:i]+each[i+1]+each[i] + each[i+2:])

        for i in range(0, total+1):
            for char in alphabet:
                edits.append(each[:i]+char+each[i:])
        return edits

    def known_edits2(self, word: str) -> List[str]:
        known = []

        edits = self.edits1(word)
        for each in edits:
            for item in self.edits1(each):
                if item in self.vocabulary and self.vocabulary[item] > 2:
                    known.append(item)
        return known

    def known(self, word: str) -> List[str]:
        known = []
        if isinstance(word, str):
            if word in self.vocabulary and self.vocabulary[word] > 3:
                known.append(word)
        elif isinstance(word, list):
            for w in word:
                if w in self.vocabulary and self.vocabulary[w] > 3:
                    known.append(w)
        return known


if __name__ == "__main__":
    pass