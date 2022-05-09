# load fasttext embeddings that are trained on wiki news. Each embedding has 300 dimensions
python -m embedding_service.server --embedding fasttext  --model pa5_data/wiki-news-300d-1M-subword.vec

# load sentence BERT embeddings that are trained on msmarco. Each embedding has 768 dimensions
python -m embedding_service.server --embedding sbert  --model msmarco-distilbert-base-v3

# load wapo docs into the index called "wapo_docs_50k"
python load_es_index.py --index_name wapo_docs_50k --wapo_path pa5_data/subset_wapo_50k_sbert_ft_filtered.jl

# use keyword from topic 363 as the query; search over the stemmed_content field from index "wapo_docs_50k" based on BM25 and compute NDCG@20
python evaluate.py --index_name wapo_docs_50k --topic_id 363 --query_type kw --use_english_analyzer --top_k 20

# use natural language from topic 363 as the query; search over the stemmed_content field from index "wapo_docs_50k" based on sentence BERT embedding reranking query and compute NDCG@20
python evaluate.py --index_name wapo_docs_50k --topic_id 363 --query_type nl --vector_name sbert_vector --top_k 20  --search_type rerank


# load fasttext embeddings that are trained on wiki news. Each embedding has 300 dimensions
python -m embedding_service.server --embedding fasttext  --model pa5_data/wiki-news-300d-1M-subword.vec --lda_model model

# load sentence BERT embeddings that are trained on msmarco. Each embedding has 768 dimensions
python -m embedding_service.server --embedding sbert  --model msmarco-distilbert-base-v3  --lda_model model

# load wapo docs into the index called "wapo_docs_50k"
python load_es_index.py --index_name wapo_docs_50k --wapo_path pa5_data/new_subset_wapo_50k_sbert_ft_filtered.jl

# use keyword from topic 363 as the query; search over the stemmed_content field from index "wapo_docs_50k" based on BM25 and compute NDCG@20
python evaluate.py --index_name wapo_docs_50k --topic_id 363 --query_type kw --use_english_analyzer --top_k 20

# use natural language from topic 363 as the query; search over the stemmed_content field from index "wapo_docs_50k" based on sentence BERT embedding reranking query and compute NDCG@20
python evaluate.py --index_name wapo_docs_50k --topic_id 363 --query_type nl --vector_name sbert_vector --top_k 20  --search_type rerank



# load fasttext embeddings that are trained on wiki news. Each embedding has 300 dimensions
python -m embedding_service.server --embedding fasttext  --model pa5_data/wiki-news-300d-1M-subword.vec

# load sentence BERT embeddings that are trained on msmarco. Each embedding has 768 dimensions
python -m embedding_service.server --embedding sbert  --model msmarco-distilbert-base-v3


# load topic embeddings that are trained on msmarco. Each embedding has 768 dimensions
python -m embedding_service.server --embedding topic  --model model


# load wapo docs into the index called "wapo_docs_50k"
python load_es_index.py --index_name wapo_docs_50k --wapo_path pa5_data/subset_wapo_50k_sbert_ft_lda_filtered.jl

# use keyword from topic 363 as the query; search over the stemmed_content field from index "wapo_docs_50k" based on BM25 and compute NDCG@20
python evaluate.py --index_name wapo_docs_50k --topic_id 363 --query_type kw --use_english_analyzer --top_k 20

# use natural language from topic 363 as the query; search over the stemmed_content field from index "wapo_docs_50k" based on sentence BERT embedding reranking query and compute NDCG@20
python evaluate.py --index_name wapo_docs_50k --topic_id 363 --query_type nl --vector_name sbert_vector --top_k 20  --search_type rerank