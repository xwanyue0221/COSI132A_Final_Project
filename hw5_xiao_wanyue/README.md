# Assignment Info
COSI132A Information Retrieval Spring 2022 - Final Project: Analysis on Different Ways of Applying NLP on Search Engine

## Description
* This is documentation of the final project for course COSI 132A Information Retrieval. 
* A TREC 2018 core corpus subset and twelve TREC topics with relevance judgments will be used for system development and result evaluation.
* In this project, we will provide example code for:
  - Populating and querying a corpus using ES
  - Implementing NDCG (normalized discounted cumulative gain) evaluation metric
  - Experimenting with “semantic” indexing and searching using simSCE paragraph embedding and topic modeling embedding
* The system consists of several key components: 
  - Displaying the similarity score between the search query and each of the documents
  - Indexing the corpus into ES with default standard analyzer and English analyzer for the text fields
  - Integrate ES into the Flask service for interactive search. Beside the traditional lexical search, the system should also allow the user to select the text representation to use for searching.
  - Evaluate the performance of 12 provided TREC query pairs using NDCG. For each of the 12 query pairs, produce a result table along with a brief analysis.


## Team Member Contribution
The team members include Capo Wang, Xiya Guan, Wanyue Xiao. Each of the team member equally contributed to this project.

**Capo Wang**: Unsupervised- and unsupervised simSCE Embedding with corresponding result analysis <br>
**Xiya Guan**: Topic Modeling Feature Embedding with corresponding result analysis <br>
**Wanyue Xiao**: Front End Web Development, "Did you mean" feature, Sorting Option, Time Filtering Option <br>


## Dataset
A larger subset of TREC 2018 core corpus that has already been processed. Specifically, each document has the following fields:

| variable      | Description                                                      |
| ------------- | ---------------------------------------------------------------- |
| doc_id        | original document id from the jsonline file                      |
| title         | article title                                                    |
| author        | article authors                                                  |
| content       | main article content (HTML tags removed)                         |
| date          | publish date in the format “yyyy/MM/dd”                          |
| annotation    | annotation for its relevance to a topic                          |
| ft_vector     | fastText embedding of the content                                |
| sbert_vector  | Sentence BERT embedding of the content                           |
| simSCE        | ----------------------------------                               |
| topic_feature | ----------------------------------                               |

**Notes**:
* For the annotation field, the value is stored as the format of topic_id-relevance. The relevance can be either 0, 1 or 2, which represents irrelevant, relevant or very relevant.
* The topic id can be mapped to the query pairs in the file pa5_data/pa5_queries.json.
* If the annotation field is empty, it can be considered that this document is irrelevant to any topics.

## Getting Started
### 1. Dependencies
This repository is Python-based, and **Python 3.9** is recommended. The dependencies include JSON, Flask, DateTime, re, elasticsearch, elasticsearch-dsl, sentence-transformers, flask, numpy, and pyzmq. 

### 2. First-time Running
All dependencies are listed in the requirement.txt file. Anyone who wishes to run this project on a local environment could install these packages using the command: <code>pip3 install -r requirements.txt </code>. Before running this project in the terminal, the user shall be aware that all the required packages listed above shall be properly installed or upgraded to the latest version. 

### 3. Download all Necessary Datasets
The data directory should contain the following files.
```
data
├── pa5_queries.json
├── ideal_relevance.json
├── subset_wapo_50k_sbert_ft_filtered.jl
├── topics2018.xml
└── wiki-news-300d-1M-subword.vec
```
You need to download the pretrained fastText embedding on wiki news and put it into data folder. You can click this [link](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip) to download. 

### 4. Activate Elasticsearch Basics
You can  can click this [link](https://www.elastic.co/downloads/past-releases#elasticsearch) to download ES. Make sure you are choosing Elasticsearch 7.10.2. 

To start the ES engine:
```shell script
cd elasticsearch-7.10.2/
./bin/elasticsearch
```
To test your ES is running, open http://localhost:9200/ in your browser. You should be able to see the health status of your ES instance with the version number, the name and more. **Note that you should keep ES running in the backend while you are building and using your IR system.**

### 5. Activate Embedding Service
Load fasttext embeddings that are trained on wiki news. Each embedding has 300 dimensions
```shell script
python -m embedding_service.server --embedding fasttext  --model pa5_data/wiki-news-300d-1M-subword.vec
```

Load sentence BERT embeddings that are trained on msmarco. Each embedding has 768 dimensions
```shell script
python -m embedding_service.server --embedding sbert  --model msmarco-distilbert-base-v3
```

Load simCSE Embedding into the index called "    "
```shell script
python 
```

Load topic feature embeddings into the index called "      "
```shell script
python 
```

Load wapo docs into the index called "wapo_docs_50k"
```shell script
python load_es_index.py --index_name wapo_docs_50k --wapo_path pa5_data/subset_wapo_50k_sbert_ft_filtered.jl
```

**Note that you should keep all these shells running in the backend while you are building and using your IR system.**


### 6. Running the Programs
The user shall follow the following step to run this program in the local environment. Run <code> python hw5.py </code> in the environment and type http://127.0.0.1:5000/ in browser to view the web application. 

For Evaluation: 
Change ```TOPIC_ID``` to the topic ID you want to evaluate.
```shell
sh scirpts.sh
```

## Evaluation Table for the 12 Example Queries

| Topic: 336 | Keywords       | Natural Language | Topic: 363 | Keywords       | Natural Language | 
| ---------- | -------------- | ---------------- | ---------- | -------------- | ---------------- |
| Vector     | 0.3044         | 0.1148           | Vector     | 0.7693         | 0.1124           |
| Rerank     | 0.2792         | 0.1328           | Rerank     | 0.7443         | 0.1361           |


| Topic: 397 | Keywords       | Natural Language | Topic: 408 | Keywords       | Natural Language | 
| ---------- | -------------- | ---------------- | ---------- | -------------- | ---------------- |
| Vector     | 0.2876         | 0.2080           | Vector     | 0.5907         | 0.5571           |
| Rerank     | 0.2591         | 0.2576           | Rerank     | 0.4438         | 0.3965           |


| Topic: 433 | Keywords       | Natural Language | Topic: 439 | Keywords       | Natural Language | 
| ---------- | -------------- | ---------------- | ---------- | -------------- | ---------------- |
| Vector     | 0.1617         | 0.0174           | Vector     | 0.1094         | 0.0192           |
| Rerank     | 0.1662         | 0.0174           | Rerank     | 0.0730         | 0.0253           |


| Topic: 442 | Keywords       | Natural Language | Topic: 690 | Keywords       | Natural Language | 
| ---------- | -------------- | ---------------- | ---------- | -------------- | ---------------- |
| Vector     | 0.3204         | 0.1877           | Vector     | 0.1638         | 0.0              |
| Rerank     | 0.3048         | 0.1699           | Rerank     | 0.1681         | 0.0              |


| Topic: 805 | Keywords       | Natural Language | Topic: 806 | Keywords       | Natural Language | 
| ---------- | -------------- | ---------------- | ---------- | -------------- | ---------------- |
| Vector     | 0.2821         | 0.1407           | Vector     | 0.8133         | 0.5702           |
| Rerank     | 0.2741         | 0.1300           | Rerank     | 0.8136         | 0.4674           |


| Topic: 816 | Keywords       | Natural Language | Topic: 822 | Keywords       | Natural Language | 
| ---------- | -------------- | ---------------- | ---------- | -------------- | ---------------- |
| Vector     | 0.1980         | 0.3945           | Vector     | 0.3984         | 0.4654           |
| Rerank     | 0.1478         | 0.3806           | Rerank     | 0.3197         | 0.3916           |

**Notes**:
* The analyzer used in this evaluation is the English Analyzer.
* For the re-rank cases, the default embedding vector used for re-ranking is fast text.
* The default top k number is 20.
* One conspicuous observation is that the NDCG scores calculated based on the keyword query are consistently higher than those based on the natural language query.
* It is interesting to find that the NDCG scores of the re-rank condition are lower than those of the vector condition. 
* The NDCG scores vary significantly from topic to topic.
* Generally, the NDCG scores seem to increase along with the increase of the top k number.

## Difficulties encountered in this assignment
* Spent at least 3 hours understanding the basic operations of ElasticSearch
* Spent 12 hours in total on this assignment
* Despite the 3 to 5 hours spent on learning and implementing elastic search in the Python environment, most of the time was spent on debugging
* It is confusing how to calculate the NDCG score. The formula indicates that the denominator shall be the DCG score of the ideal relevance for the search query. However, it is confusing about the concept of getting ideal relevance. Shall it be the sorted result of the actual relevance list or the top k elements from the gold relevance list extracted from the ideal_relevance.json file?

## Authors
* The lecturer of COSI132A provided the home template. 
* The rest functions and templates were created by Wanyue Xiao independently.

## Testing
###  TREC Topic for Evaluation: tunnel injury disaster
The evaluation for the key words of the topic #363 will be used for testing and demonstration.

```xml
<top>
<num> Number: 363 </num>
<title>
transportation tunnel disasters 
</title>
<desc> Description:
What disasters have occurred in tunnels used for transportation?  
</desc>
<narr> Narrative
A relevant document identifies a disaster in a tunnel used for trains, motor vehicles, or people. Wind tunnels and tunnels used for wiring, sewage, water, oil, etc. are not relevant. The cause of the problem may be fire, earthquake, flood, or explosion and can be accidental or planned. Documents that discuss tunnel disasters occurring during construction of a tunnel are relevant if lives were threatened.  
</narr>
</top>
```

### Output
In the following table, the evaluation scores for the example query with different combinations have been displayed. The number of the retrieved document was 20. 


| Evaluation | BM25+Standard  | BM25+English | SentenceBERT+English  | Rerank SentenceBERT+English  |
| ---------- | -------------- | ------------ | --------------------- | ---------------------------- |
| AP         | 0.84557        | 0.88460      | 0.72598               | 0.75250                      |
| Precision  | 0.85000        | 0.75000      | 0.70000               | 0.75000                      |
| NDCG@20    | 0.79794        | 0.76928      | 0.66862               | 0.63404                      |
