import os
import sys

from .inverted_index import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies

class HybridSearch:
    def __init__(self, documents: list[dict] = []) -> None:
        if documents is None or len(documents) == 0:
            documents = load_movies()
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        idx = InvertedIndex()
        idx.build_index_if_not_exists()
        self.idx = idx

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

    def hybrid_score(self, bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
        return alpha * bm25_score + (1 - alpha) * semantic_score

    def weighted_search(self, query: str, alpha: float = 0.5, limit: int=5):
        wlimit = limit * 500
        bm25 = self._bm25_search(query=query, limit=wlimit)
        chunks = self.semantic_search.search_chunks(query=query, limit=wlimit, documents=self.documents)
        bm25 = normalize_min_max_dicts(bm25)
        chunks = normalize_min_max_dicts(chunks)
        res = []
        for item in bm25:
            d = {"id": item["id"], "title": item["title"], "description": item["description"], "bm25_score": item["score"]}
            for c in chunks:
                if c["id"] == item["id"]:
                    d["semantic_score"] = c["score"]
                    break
            if d.get("semantic_score", None):
                hscore = self.hybrid_score(d["bm25_score"], d["semantic_score"], alpha)
                d["hybrid_score"] = hscore
            else:
                d["hybrid_score"] = d["bm25_score"]
            res.append(d)
        topX = sorted(res, key=lambda x: x["hybrid_score"], reverse=True)[:limit]
        return topX


def weighted_search(query: str, alpha: float = 0.5, limit: int=5):
    hs = HybridSearch()
    res = hs.weighted_search(query=query, alpha=alpha, limit=limit)
    for i in range(len(res)):
        print(f"{i+1}. {res[i]['title']}")
        print(f"  Hybrid Score: {res[i]['hybrid_score']:.3f}")
        print(f"  BM25: {res[i]['bm25_score']:.3f}, Semantic: {res[i]['semantic_score']:.3f}")
        print(f"{res[i]['description'][:100]}")


def normalize_min_max_dicts(scores: list[dict]) -> list[dict]:
    if len(scores) == 0:
            return []
    if len(scores) == 1:
        scores[0]["score"] = 1.0
        return scores

    min = float("inf")
    max = float("-inf")
    
    for score in scores:
        if score["score"] > max:
            max = score["score"]
        if score["score"] < min:
            min = score["score"]

    for score in scores:
        s = 1.0
        if min < max:
            s = (score["score"] - min) / (max - min)
        score["score"] = s

    return scores


def normalize_min_max(scores: list[float]) -> list[float]:
    if len(scores) == 0:
        return []
    if len(scores) == 1:
        return [1.0]
    
    min = float("inf")
    max = float("-inf")

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
    