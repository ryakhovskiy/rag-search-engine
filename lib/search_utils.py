import json
import string
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def search_from_index(query: str, index: dict, docmap: dict, limit: int = 5) -> list[dict]:
    stopwords = load_stopwords()
    query_tokens = tokenize_text(query, stopwords, stemmer)
    docs: list[str] = []
    for token in query_tokens:
        doc_ids = index.get(token, [])
        print(f"for token {token} found {len(doc_ids)} movies")
        if len(doc_ids) > 0:
            docs.extend(doc_ids)
        if len(docs) >= limit:
            break
    res: list[str] = []
    print(f"found {len(docs)} movies in total...")
    for id in docs:
        desc = docmap[id]
        movie = f"{id}: {desc}"
        res.append(movie)
    return res


def search_command(query: str, limit: int = 5) -> list[dict]:
    stopwords = load_stopwords()
    movies = load_movies()
    results = []
    for movie in movies:
        title = movie["title"]
        query_tokens = tokenize_text(query, stopwords, stemmer)
        title_tokens = tokenize_text(title, stopwords, stemmer)
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
    return results


def load_movies(path: str = "data/movies.json"):
    print(f"loading movies from the file {path}...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        print(f"movies loaded...")
        return data["movies"]

def normalize_tokens(tokens: list[str], stopwords: list[str], stemmer) -> list[str]:
    res: list[str] = []
    for token in tokens:
        if token not in stopwords:
            token = stemmer.stem(token)
            res.append(token)
    return res

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str, stopwords: list[str] = []) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = [token for token in tokens if len(token) > 0]
    if len(stopwords) > 0:
        return normalize_tokens(valid_tokens, stopwords, stemmer)
    else:
        return valid_tokens


def load_stopwords(path: str = "data/stopwords.txt") -> list[str]:
    res: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
        lines = data.splitlines()
        for line in lines:
            line = preprocess_text(line)
            res.extend(tokenize_text(line))
        return res

def stem_term(term: str) -> str:
    return stemmer.stem(term)

def split_text_to_words(text) -> list[str]:
    if not text or len(text.strip()) == 0:
        return []
    return text.split(" ")


def split_text_to_sentences(text) -> list[str]:
    if not text or len(text) == 0:
        return []
    import re
    text = text.strip()
    if len(text) == 0:
        return []
    regex = r"(?<=[.!?])\s+"
    regexed = re.split(regex, text)
    if len(regexed) == 1 and regexed[0][-1:] not in ('?', '.', '!'):
        return text
    else:
        return regexed


def chunk(words: list[str], chunk_size: int = 200, overlap: int = 0) -> list[str]:
    return chunk_with_split(split_text_to_words, " ".join(words), chunk_size=chunk_size, overlap=overlap)


def chunk_with_split(split_function, text: str, chunk_size: int = 200, overlap: int = 0) -> list[str]:
    if not text or len(text.strip()) == 0:
        return []
    if chunk_size == 0:
        return [text]
    if overlap >= chunk_size:
        raise ValueError("overlap cannot be larger or equal to chunk_size")
    words = split_function(text)
    res = []
    chunk = []

    counter = 0
    i = 0
    while i < len(words):
        word = words[i].strip()
        if len(word) == 0:
            i += 1
            continue
        if counter == chunk_size:
            chk = " ".join(chunk).strip()
            if len(chk) != 0:
                res.append(chk)
            chunk = chunk[-overlap:] if overlap > 0 else []  # seed overlap into next chunk
            counter = len(chunk)
        chunk.append(word)
        counter += 1
        i += 1
    if len(chunk) > 0:
        chk = " ".join(chunk).strip()
        if len(chk) != 0:
            res.append(chk)
    return res
    
