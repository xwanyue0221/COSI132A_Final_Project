#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ast
import argparse
from datetime import datetime
# from typing import Dict, Tuple
from flask import Flask, render_template, request
from elasticsearch_dsl import Search
from elasticsearch import Elasticsearch
from elasticsearch_dsl.query import Ids
from elasticsearch_dsl.connections import connections
from evaluate import get_response, get_score
from spell_corrector import SpellCorrector

app = Flask(__name__)
es = Elasticsearch()
connections.create_connection(hosts=["localhost"], timeout=100, alias="default")
sc = SpellCorrector()
page_limit = 8


# home page
@app.route("/")
def home():
    return render_template("test.html")


# back to home page
@app.route("/home", methods=["POST"])
def back_to_home():
    return render_template("test.html")


# result page
@app.route("/results", methods=["POST"])
def results():
    """
    result page
    :return: a json object including all the articles whose titles include the users' search queries
    """

    query_text = request.form["query"]  # Get the raw user query from home page
    page_num = int(request.form['page_num'])  # Get the page number from home page
    custom_date_top = request.form['true_date_top'] if 'true_date_top' in request.form else None
    custom_date_bottom = request.form['true_date_bottom'] if 'true_date_bottom' in request.form else None
    if query_text == "":
        return home()

    sort_type = request.form['true_sorting'] if 'true_sorting' in request.form else 'relevance'
    analyzer_type = request.form['true_analyzer'] if 'true_analyzer' in request.form else 'english_analyzer'
    embed_type = request.form['true_embedding'] if 'true_embedding' in request.form else 'bm25'
    search_type = 'vector' if embed_type=='bm25' else 'rerank'
    if args.debug:
        print(analyzer_type)
        print(embed_type)
        print(sort_type)
        print(custom_date_top, custom_date_bottom)
        print()

    english_analyzer = (analyzer_type == "english_analyzer")
    response = get_response(args.index_name, query_text, english_analyzer, search_type, embed_type, args.top_k, args.debug)
    doc_result = [(hit.meta.id, round(hit.meta.score,4), hit.title, hit.content[:200]+'......', hit.date) for hit in response]
    if sort_type == "date":
        doc_result.sort(key = lambda x: x[4])

    print(type(custom_date_top), len(custom_date_top.strip()))
    if custom_date_top is not None and len(custom_date_top.strip()) > 0:
        try:
            start_date = datetime.strptime(custom_date_top.strip(), '%Y/%m/%d')
            doc_result = [each for each in doc_result if datetime.strptime(each[4], '%Y/%m/%d') >= start_date]
        except ValueError as e:
            print('Value Error')
    if custom_date_bottom is not None and len(custom_date_bottom.strip()) > 0:
        try:
            end_date = datetime.strptime(custom_date_bottom.strip(), '%Y/%m/%d')
            doc_result = [each for each in doc_result if datetime.strptime(each[4], '%Y/%m/%d') <= end_date]
        except ValueError as e:
            print('Value Error')

    if args.debug:
        print(args.top_k, query_text)

    query_token = query_text.lower().split(" ")
    recommend = []
    changed = 0

    for each in query_token:
        if sc.correct(each) == each:
            recommend.append(each)
        else:
            changed = 1
            recommend.append(sc.correct(each))
    if args.debug: print(recommend)
    recommend = ' '.join(recommend)

    doc_json ={"page_limit":page_limit, "query_text":str(query_text), "page_num":int(page_num), "doc_results":doc_result, "changed":changed,
               "sort": sort_type, "total_number":len(doc_result), "analyzer":analyzer_type, "embedding": embed_type, "spell_correct":recommend,
               "start_date":custom_date_top.strip(), "end_date":custom_date_bottom.strip()}
    return render_template("results.html", data=doc_json)


# "next page" to show more results
@app.route("/results/<int:page_id>", methods=["POST"])
def next_page(page_id):
    """
    "next page" to show more results
    :param page_id: a integer which represents the number of web page users is browsing at
    :return: a json object including all the articles whose titles include the users' search queries
    """
    query_text = request.form["query"]  # Get the raw user query from home page
    total_number=int(request.form["total_number"])
    sort_type = str(request.form["sort"])
    analyzer_type = str(request.form["analyzer"])
    embed_type = str(request.form["embedding"])
    start_date = str(request.form["true_date_top"]).strip()
    end_date = str(request.form["true_date_bottom"]).strip()
    doc_results = ast.literal_eval(request.form["doc_results"])

    if len(doc_results) == 0:
        doc_json ={"page_limit":page_limit,
                   "query_text":str(query_text),
                   "page_num":int(page_id),
                   "doc_results":doc_results,
                   "total_number":0,
                   "sort": sort_type,
                   "start_date": start_date,
                   "end_date": end_date,
                   "analyzer":analyzer_type,
                   "embedding": embed_type}
        return render_template('results.html', data=doc_json)
    else:
        doc_json ={"page_limit":page_limit,
                   "query_text":str(query_text),
                   "page_num":int(page_id),
                   "doc_results":doc_results,
                   "total_number":total_number,
                   "sort": sort_type,
                   "start_date": start_date,
                   "end_date": end_date,
                   "analyzer":analyzer_type,
                   "embedding": embed_type}
        return render_template('results.html', data=doc_json)


# document page
@app.route("/doc_data/<int:doc_id>")
def doc_data(doc_id):
    if args.debug: print(doc_id)

    search = Search(using="default", index=args.index_name).query(Ids(values=[doc_id]))
    results = search.execute()
    doc_result = [(hit.title, hit.author, hit.date, hit.content) for hit in results][0]

    doc_content ={"title":str(doc_result[0]), "author":str(doc_result[1]), "date":str(doc_result[2]), "content": str(doc_result[3])}
    return render_template("doc.html", data=doc_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elasticsearch IR system") # creating arguments
    parser.add_argument("--index_name", required=False, type=str, default="wapo_docs_50k", help="name of the ES index")
    parser.add_argument("--top_k", required=False, type=int, default=10000, help="evaluate on top k ranked documents")
    parser.add_argument("--debug", action='store_true', help="debug mode activated")
    args = parser.parse_args()
    app.run(debug=True, port=5000)
