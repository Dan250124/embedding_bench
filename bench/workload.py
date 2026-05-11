"""Test data loader."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Generator


@dataclass
class RerankerPair:
    query: str
    documents: list[str]


@dataclass
class Corpus:
    short: list[str]
    medium: list[str]
    long: list[str]
    reranker_pairs: list[RerankerPair]


def load_corpus(path: str | Path) -> Corpus:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return Corpus(
        short=data["short"],
        medium=data["medium"],
        long=data["long"],
        reranker_pairs=[
            RerankerPair(query=p["query"], documents=p["documents"])
            for p in data["reranker_pairs"]
        ],
    )


def get_batches(
    items: list, batch_size: int
) -> Generator[list, None, None]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
