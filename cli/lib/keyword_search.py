from collections import defaultdict
import pickle

from .search_utils import CACHE_PATH, load_movies, load_stopwords, DEFAULT_SEARCH_LIMIT, Movie
from nltk.stem import PorterStemmer
import string

STOPWORDS = set(load_stopwords())


class InvertedIndex():
    def __init__(self) -> None:
        self.index: dict[str, set] = defaultdict(set)
        self.docmap: dict[int, Movie] = {}
        self.index_path = CACHE_PATH / "index.pkl"
        self.docmap_path = CACHE_PATH / "docmap.pkl"

    def __add_document(self, doc_id, text) -> None:
        tokenized_text = tokenize_text(text)
        for token in tokenized_text:
            self.index[token].add(doc_id)

    def get_documents(self, term) -> list[int]:
        return sorted(self.index.get(term, []))

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

    def load(self) -> None:
        with open(CACHE_PATH / "index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open(CACHE_PATH / "docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)


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


def build_command() -> None:
    inv_idx = InvertedIndex()
    inv_idx.build()
    inv_idx.save()


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
