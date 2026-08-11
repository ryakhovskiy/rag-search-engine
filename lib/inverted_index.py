from .search_utils import tokenize_text
from .search_utils import load_movies
from .search_utils import load_stopwords
from .search_utils import stem_term
from collections import Counter

import math
import pickle
import os.path

class InvertedIndex:

    BM25_K1 = 1.5 # to calculate saturated term frequency
    BM25_B = 0.75 # to calculate doc length normalization
    CACHE_DIR = "cache"
    __index_file_path = os.path.join(CACHE_DIR, "index.pkl")
    __docmap_file_path = os.path.join(CACHE_DIR, "docmap.pkl")
    __term_frequencies_file_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
    __doc_lengths_file_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __init__(self):
        self.index: dict[str, list[int]] = dict() # token -> document_ids
        self.docmap: dict[int, str] = dict() # document_id -> tokens
        self.term_frequencies: dict[int, Counter] = dict() # document_id -> Counter(token->count)
        self.doc_lengths: dict[int, int] = dict()


    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        return float(sum(self.doc_lengths.values())) / len(self.doc_lengths)


    def __add_document(self, doc_id: int, token: str) -> None:
        counter: Counter = self.term_frequencies.get(doc_id, None)
        if counter is None:
            counter = Counter({token: 1})
            self.term_frequencies[doc_id] = counter
        else:
            counter[token] += 1
        doc_ids: list[int] = self.index.get(token, None)
        if doc_ids is not None:
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)
        else:
            self.index[token] = [doc_id]

        if self.doc_lengths.get(doc_id, 0) == 0:
            self.doc_lengths[doc_id] = 1
        else:
            self.doc_lengths[doc_id] += 1
    
    def get_document(self, term: str) -> list[str]:
        doc_ids = self.index.get(term, None)
        return sorted(doc_ids) if doc_ids else []


    def get_tf(self, doc_id: int, term: str) -> int:
        term = stem_term(term)
        counter = self.term_frequencies.get(doc_id, None)
        if counter is None:
            return 0
        else:
            return counter[term]


    def get_idf(self, term: str) -> float:
        stopwords = load_stopwords()
        tokens = tokenize_text(term, stopwords=stopwords)
        token = tokens[0] if len(tokens) > 0 else ""
        #math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        total_doc_count = (len(self.docmap) + 1.0)
        term_match_doc_count = len(self.index.get(token, [])) + 1.0
        return math.log(total_doc_count / term_match_doc_count)


    def get_tfidf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id=doc_id, term=term)
        idf = self.get_idf(term=term)
        return tf * idf


    def get_bm25_idf(self, term: str) -> float:
        term = stem_term(term)
        #log((total_docs - df + 0.5) / (df + 0.5) + 1)
        total_docs = len(self.docmap)
        df = len(self.index.get(term, []))
        return math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)


    def get_bm25_tf(self, doc_id: int, term: str, k1: float=BM25_K1, b: float=BM25_B) -> float:
        doc_length = self.doc_lengths.get(doc_id, 0)
        length_norm = 1 - b + b * (doc_length / self.__get_avg_doc_length())

        tf = self.get_tf(doc_id=doc_id, term=term)
        saturated_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return saturated_tf


    def build(self):
        stopwords = load_stopwords()
        movies = load_movies()
        total = len(movies)
        print(f"building index for {total} movies...")
        counter = 0
        for m in movies:
            id = m["id"]
            title = m["title"]
            descr = m["description"]
            text = f"{title} {descr}"
            text_tokens = tokenize_text(text, stopwords=stopwords)
            for token in text_tokens:
                self.__add_document(id, token)
            self.docmap[id] = title
            counter += 1
            if counter % 300 == 0:
                print(f"index {counter}/{total} movies")
        print(f"index build for {total} movies")
            
    
    def save(self, index_file: str = __index_file_path, 
             docmap_file: str = __docmap_file_path, 
             term_frqncy_file: str = __term_frequencies_file_path,
             doc_lengths_file: str = __doc_lengths_file_path) -> None:
        print(f"saving the index to disk to a file {index_file}")
        with open(index_file, "wb") as f:
            pickle.dump(self.index, f)
        print(f"saving the index to disk to a file {docmap_file}")
        with open(docmap_file, "wb") as f:
            pickle.dump(self.docmap, f)
        print(f"saving the term frequencies index to disk to a file {term_frqncy_file}")
        with open(term_frqncy_file, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        print(f"saving the doc lengths index to disk to a file {doc_lengths_file}")
        with open(doc_lengths_file, "wb") as f:
            pickle.dump(self.doc_lengths, f)


    def load(self, index_file: str = __index_file_path, 
             docmap_file: str = __docmap_file_path, 
             term_frqncy_file: str = __term_frequencies_file_path,
             doc_lengths_file: str = __doc_lengths_file_path) -> None:
        if not os.path.isfile(index_file):
            print(f"index file not found {index_file}")
            return
        if not os.path.isfile(docmap_file):
            print(f"docmap file not found {docmap_file}")
            return
        if not os.path.isfile(term_frqncy_file):
            print(f"term frequency file not found {term_frqncy_file}")
            return
        if not os.path.isfile(doc_lengths_file):
            print(f"doc lengths file not found {doc_lengths_file}")
            return
        print(f"loading index from {index_file}")
        with open(index_file, "rb") as f:
            self.index = pickle.load(f)
        print(f"loading docmap from {docmap_file}")
        with open(docmap_file, "rb") as f:
            self.docmap = pickle.load(f)
        print(f"loading term frequencies from {term_frqncy_file}")
        with open(term_frqncy_file, "rb") as f:
            self.term_frequencies = pickle.load(f)
        print(f"loading doc lengths from {doc_lengths_file}")
        with open(doc_lengths_file, "rb") as f:
            self.doc_lengths = pickle.load(f)
        print("data successfully loaded")