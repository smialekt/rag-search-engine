from collections import defaultdict
import pickle

from .search_utils import CACHE_PATH, load_movies, load_stopwords, DEFAULT_SEARCH_LIMIT, Movie
from nltk.stem import PorterStemmer
import string

STOPWORDS = set(load_stopwords())


class InvertedIndex():
    def __init__(self) -> None:
        self.index: dict[str, set] = defaultdict(set)
        self.docmap: dict[str, Movie] = {}

    def __add_document(self, doc_id, text):
        tokenized_text = tokenize_text(text)
        for token in tokenized_text:
            self.index[token].add(doc_id)

    def get_documents(self, term):
        return sorted(self.index.get(term, []))

    def build(self):
        movies = load_movies()
        for m in movies:
            self.__add_document(m['id'], f"{m['title']} {m['description']}")
            self.docmap[m['id']] = m

    def save(self):
        CACHE_PATH.mkdir(parents=True, exist_ok=True)

        with open(CACHE_PATH / "index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open(CACHE_PATH / "docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Movie]:
    movies = load_movies()
    results = []
    for movie in movies:
        tokenized_query = tokenize_text(query)
        tokenized_title = tokenize_text(movie["title"])
        if has_matching_token(tokenized_query, tokenized_title):
            results.append(movie)
            if len(results) >= limit:
                break
    return results


def build_command():
    inv_idx = InvertedIndex()
    inv_idx.build()
    inv_idx.save()
    docs = inv_idx.get_documents("merida")
    print(
        f"First document for token 'merida' = {docs[0]}")


def has_matching_token(query_tokens: list[str], title_tokens: list[str]):
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True

    return False


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
