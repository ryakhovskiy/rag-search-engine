from .semantic_search import SemanticSearch 
from .semantic_search import semantic_chunk
import numpy as np
import os
import json

class ChunkedSemanticSearch(SemanticSearch):

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.documents = None
        self.chunk_embeddings_file = os.path.join(self.CACHE_DIR, "chunk_embeddings.npy")
        self.chunks_metadata_file = os.path.join(self.CACHE_DIR, "cache_metadata.json")

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        chunks: list[str] = []
        metadata: list[dict] = []
        for i in range(0, len(self.documents)):
            doc = self.documents[i]
            descr = doc["description"]
            if descr is None or len(descr.strip()) == 0:
                continue
            descr_chunks = semantic_chunk(descr, 4, 1)
            chunks.extend(descr_chunks)
            for j in range(0, len(descr_chunks)):
                metadata.append({"movie_idx": i, "chunk_idx": j, "total_chunks": len(descr_chunks)})
            if i > 0 and i % 1000 == 0:
                print(f"processed {i+1}/{len(self.documents)} documents")
        print(f"processing completed, processed {i+1}/{len(self.documents)} documents")
        print(f"encoding documents...")
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = metadata
        print(f"saving chunked embeddings to {self.chunk_embeddings_file}")
        self.save_embeddings(self.chunk_embeddings, self.chunk_embeddings_file)
        print(f"saving chunkd metadata to {self.chunks_metadata_file}")
        with open(self.chunks_metadata_file, "w") as f:
            json.dump({"chunks": metadata, "total_chunks": len(chunks)}, f, indent=2)
        print(f"processing completed, returning {len(self.chunk_embeddings)} embeddings")
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.populate_documents(documents=documents)
        if os.path.exists(self.chunk_embeddings_file) and os.path.exists(self.chunks_metadata_file):    
            with open(self.chunks_metadata_file, "r") as f:
                m = json.load(f)
                self.chunk_metadata = m["chunks"]
            with open(self.chunk_embeddings_file, "rb") as f:
                self.chunk_embeddings = np.load(f)
            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents=documents)


    def search_chunks(self, query: str, documents: list[dict], limit: int = 10):
        print("search chunk")
        self.documents = documents
        query = query.strip()
        from .semantic_search import cosine_similarity
        query_embeddings = self.generate_embedding(query)
        embeddings = self.load_or_create_chunk_embeddings(documents=self.documents)
        c_similarities = []
        for i in range(0, len(embeddings)):
            emb = embeddings[i]
            chunk_idx = i
            metadata = self.chunk_metadata[chunk_idx]
            movie_idx = metadata["movie_idx"]
            score = cosine_similarity(emb, query_embeddings)
            c_similarities.append({"chunk_idx": chunk_idx, "movie_idx": movie_idx, "score": score, "metadata": metadata})
        best_matches = dict()
        for csmlrt in c_similarities:
            movie_idx = csmlrt["movie_idx"]
            score = csmlrt["score"]
            if movie_idx not in best_matches:
                best_matches[movie_idx] = (score, csmlrt["metadata"])
            else:
                if best_matches[movie_idx][0] < score:
                    best_matches[movie_idx] = (score, csmlrt["metadata"])
        best_matches = dict(sorted(best_matches.items(), key=lambda item: item[1][0], reverse=True)[:limit])
        rest: list[dict] = []
        for doc_id in best_matches.keys():
            score = best_matches[doc_id][0]
            meta = best_matches[doc_id][1]
            movie_idx = meta["movie_idx"]
            id = self.documents[movie_idx]["id"]
            rest.append({ "id": self.documents[movie_idx]["id"], "title": self.document_map[id]["title"], "description": self.document_map[id]["description"][:100], "score": score, "metadata": meta or {}, })
        return rest



def search_chunks(query: str, limit: int = 10):
    from .search_utils import load_movies
    docs = load_movies()
    search = ChunkedSemanticSearch()
    results = search.search_chunks(query=query, limit=limit, documents=docs)
    for i in range(0, len(results)):
        res = results[i]
        print(f"\n{i+1}. {res["title"]} (score: {res["score"]:.4f})")
        print(f"   {res["description"]}...")

def embed_chunks():
    from .search_utils import load_movies
    docs = load_movies()
    search = ChunkedSemanticSearch()
    embeddings = search.load_or_create_chunk_embeddings(docs)
    print(f"Generated {len(embeddings)} chunked embeddings")