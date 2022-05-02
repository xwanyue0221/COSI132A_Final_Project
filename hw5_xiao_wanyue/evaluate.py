#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from typing import List, Any
from metrics import Score
from utils import load_topic_queries
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Match, ScriptScore, Ids, Query
from elasticsearch_dsl.connections import connections
from embedding_service.client import EmbeddingClient
import csv


def get_score(response: List[Any], topic_id: str, k: int) -> Score:
    relevance = []
    for hit in response:
        if hit.annotation == f"{topic_id}-1":
            relevance.append(1)
        elif hit.annotation == f"{topic_id}-2":
            relevance.append(2)
        else:
            relevance.append(0)

    with open("./pa5_data/ideal_relevance.json", "r") as f:
        ideal_relevance = json.load(f)[topic_id]

    S = Score.eval(relevance, ideal_relevance, k)
    return S


def generate_script_score_query(query_vector: List[float], embedding_type: str) -> Query:
    """
        Generate an ES query that match all documents based on the cosine similarity

        :param query_vector: query embedding from the encoder
        :param embedding_type: embedding type, should match the field name defined in BaseDoc ("ft_vector" or "sbert_vector")

        :return: an query object
    """
    q_script = ScriptScore(query={"match_all": {}},  # use a match-all query
                            script={"source": f"cosineSimilarity(params.query_vector, '{embedding_type}') + 1.0",
                                    "params": {"query_vector": query_vector}})
    return q_script


def re_rank(query_text: str, embedding_type: str, response: List[Any], debug: bool = False) -> Query:
    """
    The purpose of this re_rank function is to restructure .

    :param query_text: str - The query or a natural language that used to match documents from the index
    :param embedding_type: str - the embedding type specified by user, available option could be fasttext embedding and sbert embedding;
                                the default value is bm25
    :param response: List[Any] - a list of top k documents that have the highest similarity rate with the search query text
    :param debug: bool - a bool value that controls debug mode

    :return: a restructured query after embedded with user-specified embedding type
    """

    if embedding_type == "ft_vector":
        if debug: print("Re-rank query with {} embedding vector".format("fasttext"))
        encoder = EmbeddingClient(host="localhost", embedding_type="fasttext")
    elif embedding_type == "sbert_vector":
        if debug: print("Re-rank query with {} embedding vector".format("sbert"))
        encoder = EmbeddingClient(host="localhost", embedding_type="sbert")
    elif embedding_type == "simCSE_vector":
        if debug: print("Re-rank query with {} embedding vector".format("simCSE"))
        encoder = EmbeddingClient(host="localhost", embedding_type="simCSE")
    else:
        raise NotImplementedError(embedding_type)

    query_vector = encoder.encode([query_text], pooling="mean").tolist()[0] # get the query embedding and convert it to a list
    q_vector = generate_script_score_query(query_vector, embedding_type) # compute the cosine similarity score between the embeddings of query text and content text
    q_match_ids = Ids(values=[hit.meta.id for hit in response])  # get doc ids from response
    q_c = (q_match_ids & q_vector) # compound query by using logic operators on retrieved ids and query vector
    return q_c


def search(index_name: str, query_text: Query, top_k: int, debug: bool = False) -> List[Any]:
    """
        The purpose of this search function is to define a search query object and use this search object to retrieve
        documents storing in the index database.

        :param index_name: str - The index name that represents the Elasticsearch "database"
        :param query_text: str - The query or a natural language that used to match documents from the index
        :param top_k: int - an integer that represents the number of documents retrieving from the index
        :param debug: bool - a bool value that controls debug mode

        :return: a list of top k documents that have the highest similarity rate with the search query text
    """

    result = Search(using="default", index=index_name).query(query_text)[:top_k]  # initialize a query and return top k results
    response = result.execute()
    if debug:
        print("Search query:", result.to_dict())
        for hit in response:
            print(hit.meta.id, hit.meta.score, hit.title, sep="\t")
    return response


def get_response(index_name:str, query_text:str, english_analyzer:bool, search_type:str, embedding:str, k:int, debug:bool=False) -> List[Any]:
    """
    The purpose of this get_response function is use the user self-defined query_text to retrieve documents storing in the index database.

    :param index_name: str - The index name that represents the Elasticsearch "database"
    :param query_text: str - The query or a natural language that used to match documents from the index
    :param english_analyzer: bool - A bool value representing whether the user want to use english analyzer to process article's content
                                or use standard analyzer to process content
    :param search_type: str - the string representing the method user specified to use for matching, the available option could be
                                    "rerank" or "vector".
    :param embedding: str - the embedding type specified by user, available option could be fasttext embedding and sbert embedding;
                                the default value is bm25
    :param top_k: int - an integer that represents the number of documents retrieving from the index
    :param debug: bool - a bool value that controls debug mode

    :return: a list of top k documents that have the highest similarity rate with the search query text
    """

    if english_analyzer:
        if debug: print("Matching on stemmed content with english analyzer")
        q_basic = Match(stemmed_content={"query": query_text}) # match query based on stemmed content if user choose english analyzer
    else:
        if debug: print("Matching on content with standard analyzer")
        q_basic = Match(content={"query": query_text}) # match query based on content if user choose english analyzer

    if debug: print("embedding:", embedding, "  search type:", search_type, "  query text:", query_text)
    # rank documents based on the embedding type
    if search_type == "vector":
        if embedding == "bm25":
            if debug: print("Rank query with {} embedding vector".format("bm25"))
            response = search(index_name, q_basic, k, debug) # using query object to search the top k documents
        elif embedding == "ft_vector":
            if debug: print("Rank query with {} embedding vector".format("fasttext"))
            encoder = EmbeddingClient(host="localhost", embedding_type="fasttext")
            query_vector = encoder.encode([query_text], pooling="mean").tolist()[0]
            q_vector = generate_script_score_query(query_vector, embedding)
            response = search(index_name, q_vector, k)
        elif embedding == "sbert_vector":
            if debug: print("Rank query with {} embedding vector".format("sbert"))
            encoder = EmbeddingClient(host="localhost", embedding_type="sbert")
            query_vector = encoder.encode([query_text], pooling="mean").tolist()[0]
            q_vector = generate_script_score_query(query_vector, embedding)
            response = search(index_name, q_vector, k)
        elif embedding == "simCSE_vector":
            if debug: print("Rank query with {} embedding vector".format("simCSE"))
            encoder = EmbeddingClient(host="localhost", embedding_type="simCSE")
            query_vector = encoder.encode([query_text], pooling="mean").tolist()[0]
            q_vector = generate_script_score_query(query_vector, embedding)
            response = search(index_name, q_vector, k)
        else:
            raise NotImplementedError(embedding)

    # if the first ranking is based on the default bm25 and the search type was specified as "re-rank", rerank the operations
    if search_type == "rerank":
        assert query_text, f"Reranking with {embedding} can only happen if query text is not empty!"

        # using query object to search the top k documents
        if debug: print("Rank query with {} embedding vector".format("bm25"))
        response = search(index_name, q_basic, k, debug)

        if debug: print("Re-rank with {} embedding vector".format(embedding))
        rescore_query = re_rank(query_text, embedding, response, debug)  # re-rank the top k response if user specifies the embedding method
        response = search(index_name, rescore_query, k) # re-rank
    return response


def main():
    connections.create_connection(hosts=["localhost"], timeout=100, alias="default") # getting connection to the elasticsearch server
    parser = argparse.ArgumentParser(description="Elasticsearch IR system") # creating arguments
    parser.add_argument("--index_name", required=True, type=str, default="wapo_docs_50k", help="name of the ES index")
    parser.add_argument("--topic_id", type=str, default="TOPIC_ID", help="topic id number")
    parser.add_argument("--query_type", type=str, default='kw', help="use keyword or natural language query")
    parser.add_argument("--use_english_analyzer", action='store_true', help="use english analyzer for BM25 search")
    parser.add_argument("--search_type", required=False, type=str, default='vector', help="reranking or ranking with vector only")
    parser.add_argument("--vector_name", required=False, type=str, default="bm25", help="use fasttext or sbert embedding")
    parser.add_argument("--top_k", required=True, type=int, default=10000, help="evaluate on top k ranked documents")
    parser.add_argument("--debug", action='store_true', help="debug mode activated")
    args = parser.parse_args()

    # loading example queries from the pa5_queries.json file
    queries = load_topic_queries("pa5_data/pa5_queries.json")

    top_k = int(args.top_k)

    # for each of the 12 example queries, calculate the ndcg score under different conditions
    writeToCSV = True
    if writeToCSV:
        print()
        print("Start Queries Evaluation")
        print("****************"*3)
        # print()
        English_Analyzer = True
        query_topic = [k for k in queries]
        print(query_topic)
        header = ['name', 'kw', 'nl']
        sum_vector_kw_bm25 = 0
        sum_vector_nl_bm25 = 0
        sum_vector_kw_sbert = 0
        sum_vector_nl_sbert = 0
        sum_vector_kw_simCSE = 0
        sum_vector_nl_simCSE = 0
        
        sum_rerank_kw_ft = 0
        sum_rerank_nl_ft = 0
        sum_rerank_kw_sbert = 0
        sum_rerank_nl_sbert = 0
        sum_rerank_kw_simCSE = 0
        sum_rerank_nl_simCSE = 0
        


        for topic in query_topic:
            with open(f'./scores/top{top_k}_for_{topic}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                print("topic ", topic, sep="\t")

                query_text1 = queries[topic]['kw']
                query_text2 = queries[topic]['nl']

                vector_kw_bm25 = get_score(get_response(args.index_name, query_text1, English_Analyzer, "vector", 'bm25', top_k, args.debug), topic, top_k).ndcg
                vector_nl_bm25 = get_score(get_response(args.index_name, query_text2, English_Analyzer, "vector", 'bm25', top_k, args.debug), topic, top_k).ndcg
                writer.writerow(['vector_bm25', round(vector_kw_bm25, 4), round(vector_nl_bm25, 4)])
                sum_vector_kw_bm25 += vector_kw_bm25
                sum_vector_nl_bm25 += vector_nl_bm25

                vector_kw_sbert = get_score(get_response(args.index_name, query_text1, English_Analyzer, "vector", 'sbert_vector', top_k, args.debug), topic, top_k).ndcg
                vector_nl_sbert = get_score(get_response(args.index_name, query_text2, English_Analyzer, "vector", 'sbert_vector', top_k, args.debug), topic, top_k).ndcg
                writer.writerow(['vector_sbert', round(vector_kw_sbert, 4), round(vector_nl_sbert, 4)])
                sum_vector_kw_sbert += vector_kw_sbert
                sum_vector_nl_sbert += vector_nl_sbert

                vector_kw_simCSE = get_score(get_response(args.index_name, query_text1, English_Analyzer, "vector", 'simCSE_vector', top_k, args.debug), topic, top_k).ndcg
                vector_nl_simCSE = get_score(get_response(args.index_name, query_text2, English_Analyzer, "vector", 'simCSE_vector', top_k, args.debug), topic, top_k).ndcg
                writer.writerow(['vector_simCSE', round(vector_kw_simCSE, 4), round(vector_nl_simCSE, 4)])
                sum_vector_kw_simCSE += vector_kw_simCSE
                sum_vector_nl_simCSE += vector_nl_simCSE

                rerank_kw_ft = get_score(get_response(args.index_name, query_text1, English_Analyzer, "rerank", "ft_vector", top_k, args.debug), topic, top_k).ndcg
                rerank_nl_ft = get_score(get_response(args.index_name, query_text2, English_Analyzer, "rerank", "ft_vector", top_k, args.debug), topic, top_k).ndcg
                writer.writerow(['rerank_ft', round(rerank_kw_ft, 4), round(rerank_nl_ft, 4)])
                sum_rerank_kw_ft += rerank_kw_ft
                sum_rerank_nl_ft += rerank_nl_ft

                rerank_kw_sbert = get_score(get_response(args.index_name, query_text1, English_Analyzer, "rerank", "sbert_vector", top_k, args.debug), topic, top_k).ndcg
                rerank_nl_sbert = get_score(get_response(args.index_name, query_text2, English_Analyzer, "rerank", "sbert_vector", top_k, args.debug), topic, top_k).ndcg
                writer.writerow(['rerank_sbert', round(rerank_kw_sbert, 4), round(rerank_nl_sbert, 4)])
                sum_rerank_kw_sbert += rerank_kw_sbert
                sum_rerank_nl_sbert += rerank_nl_sbert

                rerank_kw_simCSE = get_score(get_response(args.index_name, query_text1, English_Analyzer, "rerank", "simCSE_vector", top_k, args.debug), topic, top_k).ndcg
                rerank_nl_simCSE = get_score(get_response(args.index_name, query_text2, English_Analyzer, "rerank", "simCSE_vector", top_k, args.debug), topic, top_k).ndcg
                writer.writerow(['rerank_simCSE', round(rerank_kw_simCSE, 4), round(rerank_nl_simCSE, 4)])
                sum_rerank_kw_simCSE += rerank_kw_simCSE
                sum_rerank_nl_simCSE += rerank_nl_simCSE
                # f.close()
        
        with open(f'./scores/average.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(['vector_bm25', round(sum_vector_kw_bm25/12, 4), round(sum_vector_nl_bm25/12, 4)])
            writer.writerow(['vector_sbert', round(sum_vector_kw_sbert/12, 4), round(sum_vector_nl_sbert/12, 4)])
            writer.writerow(['vector_simCSE', round(sum_vector_kw_simCSE/12, 4), round(sum_vector_nl_simCSE/12, 4)])
            writer.writerow(['rerank_ft', round(sum_rerank_kw_ft/12, 4), round(sum_rerank_nl_ft/12, 4)])
            writer.writerow(['rerank_sbert', round(sum_rerank_kw_sbert/12, 4), round(sum_rerank_nl_sbert/12, 4)])
            writer.writerow(['rerank_simCSE', round(sum_rerank_kw_simCSE/12, 4), round(sum_rerank_nl_simCSE/12, 4)])
            # f.close()
        print()
        print("****************"*3)
        print("Queries Evaluation End")
    else:
        ndcg_score = get_score(response, args.topic_id, top_k)
        print(f"score of {args.query_type:11s}: {ndcg_score.ap:.5f}")
        print(f"score of {args.query_type:11s}: {ndcg_score.prec:.5f}")
        print(f"score of {args.query_type:11s}: {ndcg_score.ndcg:.5f}")


if __name__ == "__main__":
    main()
