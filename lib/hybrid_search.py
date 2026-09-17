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


def rrf_search(query: str, k: int = 60, limit: int = 5, rerank_method: str = None, llm_evaluate: bool = False) -> list[dict]:
    print(f"rrf search for '{query}', k={k}, limit={limit}, rerank-method: '{rerank_method}', llm-evaluation: {llm_evaluate}")
    if rerank_method is not None and len(rerank_method) > 0:
        limit *= 5
    search = HybridSearch()
    res = search.rrf_search(query=query, k=k, limit=limit)
    print_rrf_search_results_debug_sorted(res, 'bm25_score')
    print_rrf_search_results_debug_sorted(res, 'semantic_score')
    print_rrf_search_results_debug_sorted(res, 'rrf_score')
    if rerank_method == "individual":
        counter = 1
        for item in res:
            print(f"step {counter}/{len(res)}: submitting reranking for movie {item['title']}: rrf: {item['rrf_score']}, sem_r: {item['semantic_rank']}, bm25_r: {item['bm25_rank']}")
            score = run_reranking(item, query)
            item["rerank_score"] = float(score)
            counter += 1
        res = sorted(res, key=lambda x: x["rerank_score"], reverse=True)[:limit // 5]
    if rerank_method == "batch":
        print("running batch rrf search")
        ranks = run_batch_reranking(res, query)
        for i in range(0, len(ranks)):
            res[i]["rerank_score"] = float(ranks[i])
        res = sorted(res, key=lambda x: x["rerank_score"], reverse=True)[:limit // 5]
    if rerank_method == "cross_encoder":
        pairs = list()
        for doc in res:
            pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
        from sentence_transformers import CrossEncoder
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2", device="cpu")
        ranks = cross_encoder.predict(pairs)
        for i in range(0, len(ranks)):
            res[i]["rerank_score"] = float(ranks[i])
        res = sorted(res, key=lambda x: x["rerank_score"], reverse=True)[:limit // 5]
    print_rrf_search_results_debug_sorted(res, 'rerank_score')
    if llm_evaluate:
        from .llm_client import evaluate_results
        formatted_res = format_results_for_llm_evaluation(res)
        scores = evaluate_results(query, formatted_res)
        for i, item in enumerate(res, start=0):
            print(f"{i+1}. {item['title']}: {scores[i]}/3")
    # skip printing
    # print_rrf_search_results(res)
    return res


def format_results_for_llm_evaluation(res: dict) -> list[str]:
    return [x['title'] + ' - ' + x['description'][:200] for x in res]


def print_rrf_search_results_debug_sorted(res, sort_key: str):
    sorted_res = sorted(res, key=lambda x: x.get(sort_key, 0), reverse=True)
    col_w = max((len(item['title']) for item in sorted_res), default=10) + 5
    sep = '=' * (col_w + 50)
    print(sep)
    print(f"rrf search resultset: {len(res)} items. Sorted by {sort_key}")
    for i, item in enumerate(sorted_res, start=1):
        title = f"{i}: {item['title']}:".ljust(col_w)
        bm25 = item.get('bm25_score') or 0.0
        sem  = item.get('semantic_score') or 0.0
        rrf  = item.get('rrf_score') or 0.0
        print(f"{title}  BM25: {bm25:.4f}  SEM: {sem:.4f}  RRF: {rrf:.4f}")
    print(sep)


def print_rrf_search_results(res):
    for i in range(len(res)):
        print(f"{i+1}. {res[i]['title']}")
        if "rerank_score" in res[i]:
            print(f"  Re-rank Score: {res[i]['rerank_score']:.3f}/10")
        print(f"  RRF Score: {res[i]['rrf_score']:.3f}")
        print(f"  BM25 Rank: {res[i].get('bm25_rank', 'N/A')}, Semantic Rank: {res[i].get('semantic_rank', 'N/A')}")
        print(f"{res[i]['description'][:100]}")


def run_batch_reranking(docs: list[dict], query: str) -> str:
    from .llm_client import rerank_doc_batch
    ranks = rerank_doc_batch(docs, query)
    return ranks


def run_reranking(document: dict, query: str) -> int:
    from .llm_client import rerank_doc
    counter = 0
    while counter < 3:
        rank = rerank_doc(doc=document, query=query)
        if rank.isdigit():
            return rank
        else:
            counter += 1
            print(f"llm call failed: '{rank}', trying again {counter}/3")
    return 0


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
    