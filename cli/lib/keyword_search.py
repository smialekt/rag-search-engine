from .search_utils import load_movies, load_stopwords, DEFAULT_SEARCH_LIMIT, Movie
import string

STOPWORDS = load_stopwords()


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
    return [x for x in text.split() if x and x not in STOPWORDS]
