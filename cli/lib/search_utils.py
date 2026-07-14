import json
import os
from typing import Any, TypedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data"
CACHE_PATH = PROJECT_ROOT / "cache"
MOVIES_PATH = DATA_PATH / "movies.json"
STOPWORDS_PATH = DATA_PATH / "stopwords.txt"

DEFAULT_SEARCH_LIMIT = 200


class Movie(TypedDict):
    id: int
    title: str
    description: str


def load_movies() -> list[Movie]:
    with open(MOVIES_PATH, 'r') as f:
        data = json.load(f)
    return data['movies']


def load_stopwords() -> list[str]:
    data = []
    with open(STOPWORDS_PATH, 'r') as f:
        data = f.read().splitlines()
    return data
