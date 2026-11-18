"""Initializes dense, hybrid and graph retrievers once and reuses them."""
import logging

from src.retrieval import StandardRAGRetriever, HybridRAGRetriever
from src.graph_retrieval import GraphRAGRetriever

logger = logging.getLogger(__name__)


class HybridGraphRAG:
    def __init__(
        self,
        data_path="data/processed_segments.jsonl",
        graph_path="data/cti_knowledge_graph_trusted.json",
        assertions_path="data/assertions.jsonl",
        dense_retriever=None,
        graph_retriever=None,
    ):
        logger.info("Initializing retrieval systems...")
        self.dense = dense_retriever or StandardRAGRetriever(data_path=data_path)
        self.hybrid = HybridRAGRetriever(self.dense)
        self.graph = graph_retriever or GraphRAGRetriever(
            graph_path=graph_path, assertions_path=assertions_path
        )
        logger.info("All retrieval systems are ready!")

    def retrieve(self, query, top_k=5):
        return {
            "dense": self.dense.retrieve(query, top_k=top_k),
            "hybrid": self.hybrid.retrieve(query, top_k=top_k),
            "graph": self.graph.retrieve(query, top_k=top_k),
        }
