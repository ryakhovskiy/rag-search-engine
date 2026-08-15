import numpy as np
import os

from sentence_transformers import SentenceTransformer
from torch import Tensor
from dotenv import load_dotenv
load_dotenv()




def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

class SemanticSearch:
    CACHE_DIR = "cache"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.document_map: dict = dict()
        self.embeddings = None
        self.documents = None
        self.embeddings_file = os.path.join(self.CACHE_DIR, "movie_embeddings.npy")

    def search(self, query, limit):
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embeddings = self.generate_embedding(query)
        # calculate cosine similarities
        c_similarities = []
        for emb, doc in zip(self.embeddings, self.documents):
            cs = cosine_similarity(emb, query_embeddings)
            c_similarities.append({"score": cs, "title": doc["title"], "description": doc["description"]})
        c_similarities.sort(key=lambda x: x["score"], reverse=True)
        return c_similarities[:limit]
        

    def generate_embedding(self, text: str):
        if len(text.strip()) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        embddings = self.model.encode([text])
        return embddings[0]

    def populate_documents(self, documents):
        self.documents = documents
        for doc in documents:
            id = doc["id"]
            self.document_map[id] = doc

    def build_embeddings(self, documents: list[dict[int, str]]):
        self.populate_documents(documents)
        doc_list = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
        self.save_embeddings(embeddings=self.embeddings, embeddings_file=self.embeddings_file)
        return self.embeddings

    def save_embeddings(self, embeddings, embeddings_file: str):
        with open(embeddings_file, "wb") as f:
            np.save(f, embeddings)

    def load_or_create_embeddings(self, documents):
        self.populate_documents(documents=documents)
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, "rb") as f:
                self.embeddings = np.load(f)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        self.embeddings = self.build_embeddings(documents=documents)
        return self.embeddings





def semantic_chunk(text: str, max_chunk_size: int = 4, overlap: int = 0) -> list[str]:
    from .search_utils import chunk_with_split
    from .search_utils import split_text_to_sentences
    return chunk_with_split(split_text_to_sentences, text, chunk_size=max_chunk_size, overlap=overlap)


def embed_query_text(query: str):
    search = SemanticSearch()
    query_embeddings = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {query_embeddings[:3]}")
    print(f"Shape: {query_embeddings.shape}")


def verify_embeddings():
    from .search_utils import load_movies
    docs = load_movies()
    search = SemanticSearch()
    emdgs = search.load_or_create_embeddings(docs)
    print(f"Number of docs:   {len(docs)}")
    print(f"Embeddings shape: {emdgs.shape[0]} vectors in {emdgs.shape[1]} dimensions")

def verify_model():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")

def embed_text(text: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def search(query: str, limit:int = 5) -> list[dict[float, any]]:
    from .search_utils import load_movies
    docs = load_movies()
    search = SemanticSearch()
    search.load_or_create_embeddings(docs)
    res = search.search(query=query, limit=limit)
    for r in res:
        print(r)
    return res