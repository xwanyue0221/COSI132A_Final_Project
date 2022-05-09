from typing import Iterator, Dict, Union, Sequence, Generator
from elasticsearch_dsl import Index  # type: ignore
from elasticsearch_dsl.connections import connections  # type: ignore
from elasticsearch.helpers import bulk
from es_service.doc_template import BaseDoc


class ESIndex(object):
    def __init__(self, index_name: str, docs: Union[Iterator[Dict], Sequence[Dict]]):
        """
        ES index structure
        :param index_name: the name of your index
        :param docs: wapo docs to be loaded
        """
        # set an elasticsearch connection to your localhost
        connections.create_connection(hosts=["localhost"], timeout=100, alias="default")
        self.index = index_name
        es_index = Index(self.index)  # initialize the index

        # delete existing index that has the same name
        if es_index.exists():
            es_index.delete()

        es_index.document(BaseDoc)  # link document mapping to the index
        es_index.create()  # create the index
        if docs is not None:
            self.load(docs)

    @staticmethod
    def _populate_doc(docs: Union[Iterator[Dict], Sequence[Dict]]) -> Generator[BaseDoc, None, None]:
        """
        populate the BaseDoc
        :param docs: wapo docs
        :return:
        """
        for i, doc in enumerate(docs):
            es_doc = BaseDoc(_id=i)
            es_doc.doc_id = doc["doc_id"]
            es_doc.title = doc["title"]
            es_doc.author = doc["author"]
            es_doc.content = doc["content_str"]
            es_doc.stemmed_content = doc["content_str"]
            es_doc.annotation = doc["annotation"]
            es_doc.date = doc["published_date"]
            es_doc.ft_vector = doc["ft_vector"]
            es_doc.sbert_vector = doc["sbert_vector"]
            es_doc.topic_vector = doc["topic_vector"]
            yield es_doc

    def load(self, docs: Union[Iterator[Dict], Sequence[Dict]]):
        # bulk insertion
        bulk(connections.get_connection(),
             (
                d.to_dict(
                    include_meta=True, skip_empty=False
                )  # serialize the BaseDoc instance (include meta information and not skip empty documents)
                for d in self._populate_doc(docs)
            ),
        )
