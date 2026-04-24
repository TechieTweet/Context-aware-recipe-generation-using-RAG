# retrieval/hybrid_retriever.py
# Merges BM25 sparse + dense vector results using Reciprocal Rank Fusion (RRF)
# This is the main entry point for retrieval — call hybrid_retrieve()

import pandas as pd
from config import BM25_CANDIDATES, DENSE_CANDIDATES, RRF_K, TOP_K
from retrieval.bm25_retriever import query_bm25
from retrieval.vector_store import query_dense


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """
    Merge multiple ranked lists into one using RRF.

    Each document gets score = sum(1 / (k + rank)) across all lists.
    k=60 is the standard default from the original RRF paper (Cormack et al. 2009).

    Args:
        rankings: list of ranked lists of DataFrame indices
        k: RRF constant (default 60)

    Returns:
        Single merged ranked list of DataFrame indices
    """
    scores: dict[int, float] = {}
    for ranked_list in rankings:
        for rank, doc_idx in enumerate(ranked_list):
            if doc_idx not in scores:
                scores[doc_idx] = 0.0
            scores[doc_idx] += 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def hybrid_retrieve(
    query: str,
    df: pd.DataFrame,
    bm25,
    collection,
    embedder,
    top_k: int = TOP_K,
    bm25_candidates: int = BM25_CANDIDATES,
    dense_candidates: int = DENSE_CANDIDATES,
) -> list[dict]:
    """
    Hybrid retrieval: BM25 sparse + MiniLM dense, merged with RRF.

    Args:
        query: user query string
        df: cleaned recipes DataFrame
        bm25: loaded BM25Okapi index
        collection: loaded ChromaDB collection
        embedder: loaded SentenceTransformer model
        top_k: number of final results to return
        bm25_candidates: pool size for BM25
        dense_candidates: pool size for dense search

    Returns:
        List of dicts with keys: title, ingredients, full_text, df_index
    """
    # Sparse retrieval
    bm25_indices = query_bm25(bm25, query, n_results=bm25_candidates)

    # Dense retrieval
    dense_indices = query_dense(collection, embedder, query, n_results=dense_candidates)

    # Fuse
    fused = reciprocal_rank_fusion([bm25_indices, dense_indices])
    top_indices = fused[:top_k]

    # Build result dicts
    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        results.append({
            "title":       row["title"],
            "ingredients": row["ingredients"],
            "full_text":   row["full_text"],
            "df_index":    int(idx)
        })

    return results
