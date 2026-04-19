# retrieval/bm25_retriever.py
# Builds and queries the BM25 sparse index

import re
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
import pandas as pd
from config import BM25_INDEX_PATH


def tokenize(text: str) -> list[str]:
    """Lowercase + strip punctuation tokenizer for BM25."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def build_bm25_index(df: pd.DataFrame) -> BM25Okapi:
    """
    Build a BM25 index from the document strings in df.
    Saves the index to disk.

    Args:
        df: DataFrame with a 'document' column

    Returns:
        Fitted BM25Okapi index
    """
    print("Tokenizing documents for BM25...")
    tokenized_corpus = [tokenize(doc) for doc in df["document"].tolist()]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print(f"BM25 index saved to {BM25_INDEX_PATH}")
    return bm25


def load_bm25_index() -> BM25Okapi:
    """Load a previously saved BM25 index from disk."""
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)
    print("BM25 index loaded.")
    return bm25


def query_bm25(bm25: BM25Okapi, query: str, n_results: int = 50) -> list[int]:
    """
    Run a BM25 sparse search.

    Returns:
        List of integer DataFrame indices of the top results
    """
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:n_results].tolist()
    return top_indices
