from collections import defaultdict, Counter
import pickle
import math

from .search_utils import BM25_B, CACHE_PATH, BM25_K1, load_movies, load_stopwords, DEFAULT_SEARCH_LIMIT, Movie
from nltk.stem import PorterStemmer
import string

STOPWORDS = set(load_stopwords())


class InvertedIndex():
    def __init__(self) -> None:
        self.index: dict[str, set] = defaultdict(set)
        self.docmap: dict[int, Movie] = {}
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        self.doc_lengths: dict[int, int] = {}
        self.index_path = CACHE_PATH / "index.pkl"
        self.docmap_path = CACHE_PATH / "docmap.pkl"
        self.term_frequencies_path = CACHE_PATH / "term_frequencies.pkl"
        self.doc_lengths_path = CACHE_PATH / "doc_lengths.pkl"

    def __add_document(self, doc_id, text) -> None:
        tokenized_text = tokenize_text(text)
        for token in set(tokenized_text):
            self.index[token].add(doc_id)
        self.doc_lengths[doc_id] = len(tokenized_text)
        self.term_frequencies[doc_id].update(tokenized_text)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0

        lengths = self.doc_lengths.values()
        return sum(lengths) / len(lengths)

    def get_documents(self, term) -> list[int]:
        return sorted(self.index.get(term, []))

    def get_tf(self, doc_id: int, term: str):
        return self.term_frequencies[doc_id][term]

    def get_idf(self, token: str) -> float:
        n_docs = len(self.docmap)
        n_term_match = len(self.index[token])
        return math.log((n_docs + 1) / (n_term_match + 1))

    def get_tf_idf(self, doc_id: int, token: str) -> float:
        tf = self.get_tf(doc_id, token)
        idf = self.get_idf(token)
        return tf * idf

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            self.__add_document(m['id'], f"{m['title']} {m['description']}")
            self.docmap[m['id']] = m

    def save(self) -> None:
        CACHE_PATH.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_bm25_idf(self, token: str) -> float:
        n_doc = len(self.docmap)
        n_term_match = len(self.index[token])
        return math.log((n_doc - n_term_match + 0.5) / (n_term_match + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B):
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * \
                (self.doc_lengths.get(doc_id, 0) / avg_doc_length)
        else:
            length_norm = 1
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)


def bm25_idf_command(term: str):
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(tokenize_term(term))


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B):
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, tokenize_term(term), k1, b)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Movie]:
    idx = InvertedIndex()
    idx.load()
    tokenized_query = tokenize_text(query)

    seen, results = set(), []
    for token in tokenized_query:
        matching_doc_ids = idx.get_documents(token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            results.append(idx.docmap[doc_id])
        if len(results) >= limit:
            break

    return results


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(tokenize_term(term))


def tf_idf_command(doc_id: int, term: str):
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id, tokenize_term(term))


def build_command() -> None:
    inv_idx = InvertedIndex()
    inv_idx.build()
    inv_idx.save()


def tf_command(doc_id: int, term: str) -> None:
    idx = InvertedIndex()
    idx.load()
    print(idx.get_tf(doc_id, tokenize_term(term)))


def preprocess_text(text: str) -> str:
    """Basic text preprocessing
        - removes punctuation
        - converts to lowercase
    """
    return text.translate(str.maketrans("", "", string.punctuation)).lower()


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    stemmer = PorterStemmer()
    return [stemmer.stem(x) for x in text.split() if x and x not in STOPWORDS]


def tokenize_term(term: str) -> str:
    tokenized = tokenize_text(term)
    if len(tokenized) != 1:
        raise ValueError("term must be a single token")
    return tokenized[0]
