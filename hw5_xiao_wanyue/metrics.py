"""
implementation of precision@k, averaged precison and NDCG@k
for this assignment you only need to use NDCG@k
"""
from typing import Sequence, NamedTuple, List
import warnings
import math
import numpy as np  # type: ignore


def precision(relevance: Sequence[int], k: int = 20) -> float:
    """

    :param relevance: the relevance score (0, 1, 2) of a list of documents
    :param k: top k
    :return:
    """
    relevance_len = len(relevance)
    if relevance_len < k:
        warnings.warn(
            f"sequence length is smaller than k ({k})! Reset k to maximum sequence length ({relevance_len})", SyntaxWarning,)
    top_k = relevance[:k]
    return np.count_nonzero(top_k) / k


def average_precision(relevance: Sequence[int]) -> float:
    relevant_items = np.count_nonzero(relevance)
    if not relevant_items:
        return 0.0
    return (sum(precision(relevance, k) for k, rel in enumerate(relevance, 1) if rel) / relevant_items)


def dcg(relevance: Sequence[int], k: int = 20) -> float:
    relevance_len = len(relevance)
    if relevance_len < k:
        warnings.warn(
            f"sequence length is smaller than k ({k})! Reset k to maximum sequence length ({relevance_len})", SyntaxWarning,)
    top_k = relevance[:k]
    return sum(rel / math.log2(i + 1) for i, rel in enumerate(top_k, 1))


def ndcg(relevance: Sequence[int], idea_relevance: Sequence[int], k: int = 20) -> float:
    relevance_len = len(relevance)
    if relevance_len < k:
        warnings.warn(
            f"sequence length is smaller than k ({k})! Reset k to maximum sequence length ({relevance_len})", SyntaxWarning,)
    top_k = relevance[:k]
    ideal_relevance_len = len(idea_relevance)
    if ideal_relevance_len < k:
        idea_relevance += [0] * (k - ideal_relevance_len)
    idea_relevance = idea_relevance[:k]

    try:
        return dcg(top_k, k) / dcg(idea_relevance, k)
    except ZeroDivisionError:
        return 0.0


class Score(NamedTuple):
    # simple wrapper for scores using different metrics
    ap: float
    prec: float
    ndcg: float

    @classmethod
    def eval(cls, relevance: List[int], idea_relevance: Sequence[int], top_k: int) -> "Score":
        return cls(average_precision(relevance), precision(relevance, top_k), ndcg(relevance, idea_relevance, top_k),)


if __name__ == "__main__":
    # rel = [3,3,3,2,2,1,0,0]
    # true = [3,2,3,0,1,2]
    # print(dcg(true, 6)/dcg(rel, 6))
    # print(dcg(rel, 6))
    pass