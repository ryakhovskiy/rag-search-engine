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

    def rrf_score(self, rank: int, k: int = 60) -> float:
        return 1 / (k + rank)

    def hybrid_score(self, bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
        return alpha * bm25_score + (1 - alpha) * semantic_score

    def rrf_search(self, query: str, k: int = 60, limit: int = 5) -> list[dict]:
        wlimit = limit * 500
        bm25_res = self._bm25_search(query=query, limit=wlimit)
        semantic_res = self.semantic_search.search_chunks(query=query, limit=wlimit, documents=self.documents)
        res = dict()
        for rank, item in enumerate(bm25_res, start=1):
            rrf = self.rrf_score(rank, k=k)
            res[item["id"]] = {"id": item["id"], "title": item["title"], "bm25_rank": rank, "description": item["description"], "bm25_score": item["score"], "bm25_rrf": rrf}
        for rank, item in enumerate(semantic_res, start=1):
            rrf = self.rrf_score(rank, k=k)
            if res.get(item["id"], None):
                data = res[item["id"]]
                data["semantic_rank"] = rank
                data["semantic_rrf"] = rrf
                data["semantic_score"] = item["score"]
            else:
                res[item["id"]] = {"id": item["id"], "title": item["title"], "semantic_rank": rank, "description": item["description"], "semantic_score": item["score"], "semantic_rrf": rrf}

        for id in res.keys():
            data = res[id]
            bm25_rrf = data.get("bm25_rrf", None)
            semantic_rrf = data.get("semantic_rrf", None)
            if bm25_rrf is not None and semantic_rrf is not None:
                data["rrf_score"] = bm25_rrf + semantic_rrf
            elif bm25_rrf:
                data["rrf_score"] = bm25_rrf
            else:
                data["rrf_score"] = semantic_rrf
            if "I.Q." in data["title"] or "I.Q." in data["description"]:
                print(f"---> found IQ: {data}")

        return sorted(res.values(), key=lambda x: x['rrf_score'], reverse=True)[:limit]

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
                d["hybrid_score"] = d["bm25_score"]
            else:
                d["hybrid_score"] = self.hybrid_score(d["bm25_score"], d["semantic_score"], alpha)
            res.append(d)
        topX = sorted(res, key=lambda x: x["hybrid_score"], reverse=True)[:limit]
        return topX


def rrf_search(query: str, k: int = 60, limit: int = 5) -> list[dict]:
    print(f"rrf search for '{query}', k={k}, limit={limit}")
    search = HybridSearch()
    res = search.rrf_search(query=query, k=k, limit=limit)
    for i in range(len(res)):
        print(f"{i+1}. {res[i]['title']}")
        print(f"  RRF Score: {res[i]['rrf_score']:.3f}")
        print(f"  BM25 Rank: {res[i].get('bm25_rank', 'N/A')}, Semantic Rank: {res[i].get('semantic_rank', 'N/A')}")
        print(f"{res[i]['description'][:100]}")
    return res

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
    