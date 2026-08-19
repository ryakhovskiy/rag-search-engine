import os
import sys

from .inverted_index import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize_min_max(scores: list[float]) -> list[float]:
    if len(scores) == 0:
        return []
    if len(scores) == 1:
        return [1.0]
    
    min = sys.float_info.max
    max = sys.float_info.min

    for score in scores:
        if score > max:
            max = score
        if score < min:
            min = score

    if min == max:
        return [1.0] * len(scores)

    res = []
    for score in scores:
        res.append((score - min) / (max - min))
    return res
    