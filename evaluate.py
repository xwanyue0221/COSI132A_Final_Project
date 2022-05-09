#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue April 20 2021
@author: Xiaoyu Lu
@modiefied by:Yunjing Lee
"""
import argparse
from typing import List, Any
from elasticsearch_dsl import Search
from metrics import Score
from utils import parse_wapo_topics
from elasticsearch_dsl.query import Match, ScriptScore, Ids, Query
from elasticsearch_dsl.connections import connections
from embedding_service.client import EmbeddingClient
import csv


def generate_script_score_query(query_vector: List[float], vector_name: str) -> Query:
    """
    generate an ES query that match all documents based on the cosine similarity
    :param query_vector: query embedding from the encoder
    :param vector_name: embedding type, should match the field name defined in BaseDoc ("ft_vector" or "sbert_vector")
    :return: an query object
    """
    q_script = ScriptScore(
        query={"match_all": {}},  # use a match-all query
        script={  # script your scoring function
            "source": f"cosineSimilarity(params.query_vector, '{vector_name}') + 1.0",
            # add 1.0 to avoid negative score
            "params": {"query_vector": query_vector},
        },
    )
    return q_script


def search(index: str, query: Query, k: int) -> List[Any]:
    s = Search(using="default", index=index).query(query)[
        :k
        ]  # initialize a query and return top five results
    response = s.execute()
    # relevance = []
    # for hit in response:
    #     # print(
    #     #     hit.meta.id, hit.meta.score, hit.annotation, hit.title, sep="\t"
    #     # )  # print the document id that is assigned by ES index, score and title
    #     relevance.append(1 if hit.annotation in topic_rels else 0)
    return response


def _rerank_query(query_text: str, embed_methods: str, response: List[Any]) -> Query:
    """
    Created: Xiaoyu Lu
    Modified: Yonglin Wang
    """
    if embed_methods == "ft_vector":
        encoder = EmbeddingClient(host="localhost", embedding_type="fasttext")
    elif embed_methods == "sbert_vector":
        encoder = EmbeddingClient(host="localhost", embedding_type="sbert")
    elif embed_methods == "lf_vector":
        encoder = EmbeddingClient(host="localhost", embedding_type="longformer")
    else:
        raise NotImplementedError(embed_methods)

    query_vector = encoder.encode([query_text], pooling="mean").tolist()[
        0
    ]  # get the query embedding and convert it to a list

    # q_vector: cosine similarity between the embeddings of query text and content text.
    q_vector = generate_script_score_query(
        query_vector, embed_methods
    )  # custom query that scores documents based on cosine similarity

    # get doc ids from response
    q_match_ids = Ids(values=[hit.meta.id for hit in response])
    # print(q_match_ids)
    q_c = (
            q_match_ids & q_vector
    )  # compound query by using logic operators on retrieved ids and query vector

    return q_c


def get_response(index_name: str,
                 query_text: str,
                 custom_analyzer: bool,
                 embed_methods: str = "bm25",
                 k: int = 20,
                 kw_query: str = "") -> List[Any]:
    reranking = (embed_methods != "bm25")

    # get top k query response
    top_k_query = kw_query if kw_query else query_text
    assert top_k_query
    if custom_analyzer:
        q_basic = Match(
            custom_content={"query": top_k_query}
        )
    else:
        # q_basic: default analyser or custom analyzer
        q_basic = Match(
            content={"query": top_k_query}
        )  # a query that matches text in the content field of the index, using BM25 as default
    response = search(
        index_name, q_basic, k
    )  # search, change the query object to see different results

    # rerank top k response
    if reranking:
        # rerank ONLY if query text is not empty prevent fasttext NaN score problem
        assert query_text, f"Reranking with {embed_methods} can only happen if query text is not empty!"
        # embedding: fasttext or sentence bert
        q_c = _rerank_query(query_text, embed_methods, response)
        response = search(
            index_name, q_c, k,
        )  # re-ranking

    return response


def get_score(response: List[Any], topic_id: str, k: int) -> Score:
    # option 1
    # topic_rels = [f"{topic_id}-1", f"{topic_id}-2"]
    # relevance = [1 if hit.annotation in topic_rels else 0 for hit in response]
    # option 2
    relevance = []
    for hit in response:
        if hit.annotation == f"{topic_id}-1":
            relevance.append(1)
        elif hit.annotation == f"{topic_id}-2":
            relevance.append(2)
        else:
            relevance.append(0)
    # print(relevance)
    S = Score.eval(relevance, k)
    return S


def main():
    connections.create_connection(hosts=["localhost"], timeout=100, alias="default")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index_name", required=True, type=str, help="name of the index"
    )
    parser.add_argument(
        "--topic_id", required=True, type=str, help="number of the topic id"
    )
    parser.add_argument(
        "--query_type", required=True, type=str, help="Option[title,description,narration]"
    )
    parser.add_argument(
        "--vector_name", required=False, type=str, default="bm25",
        help="Option[sbert_vector, fasttext_vector, lf_vector]"
    )

    parser.add_argument(
        "-u", action='store_true', help="use custom_content field"
    )

    parser.add_argument(
        "--top_k", required=True, type=int, default=20, help="compute NDCG@k"
    )

    args = parser.parse_args()

    mapping = parse_wapo_topics("data/topics2018.xml")
    title, description, narration = mapping[args.topic_id]
    if args.query_type == "title":
        query_text = title
    elif args.query_type == "description":
        query_text = description
    elif args.query_type == "narration":
        query_text = narration
    else:
        raise ValueError
    # print(f"query_text: {query_text}")

    custom_analyzer = False
    if args.u: custom_analyzer = True
    k = int(args.top_k)
    response = get_response(args.index_name, query_text, custom_analyzer, args.vector_name, k)
    ndcg_score = get_score(response, args.topic_id, k).ndcg
    print(f"score of {args.query_type:11s}: {ndcg_score:.5f}")


if __name__ == "__main__":
    main()
