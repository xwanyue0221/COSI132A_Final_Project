# open elastic search service
#########Keep Commenting Out##########
# $ cd elasticsearch-7.10.2/ 
# $ ./bin/elasticsearch

# # load fasttext embeddings that are trained on wiki news. Each embedding has 300 dimensions
# python -m embedding_service.server --embedding fasttext  --model pa5_data/wiki-news-300d-1M-subword.vec

# # load sentence BERT embeddings that are trained on msmarco. Each embedding has 768 dimensions
# python -m embedding_service.server --embedding sbert  --model msmarco-distilbert-base-v3

# # LOAD enbeddings that trained on unsupervised simCSE
# python -m embedding_service.server --embedding simCSE  --model princeton-nlp/unsup-simcse-bert-base-uncased

# # load wapo docs into the index called "wapo_docs_50k"
# python load_es_index.py --index_name wapo_docs_50k --wapo_path pa5_data/update_wapo.jl


# calculate results 
python evaluate.py --index_name wapo_docs_50k --use_english_analyzer --top_k 20
