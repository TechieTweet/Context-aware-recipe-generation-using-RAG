# evaluation/ragas_eval.py
# RAGAS-style evaluation pipeline
# Computes: Faithfulness, Answer Relevance, Contextual Precision, Contextual Recall
# Uses cosine similarity — no external API needed

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import EVAL_RESULTS_PATH, EMBEDDING_MODEL
import torch


# ── Metric implementations ────────────────────────────────────

def compute_faithfulness(answer: str, retrieved_docs: list[str], available_ingredients: list[str] = []) -> float:
    """
    Faithfulness: fraction of answer ingredients that appear in retrieved docs
    or in the user's stated available ingredients.

    Score = non-hallucinated ingredients / total ingredients in answer.
    Target: >= 0.70
    """
    # Extract rough ingredient tokens from answer
    answer_lower = answer.lower()
    # Use ingredients section if present
    if "ingredients:" in answer_lower:
        ing_section = answer_lower.split("ingredients:")[1].split("instructions:")[0]
    else:
        ing_section = answer_lower

    answer_tokens = set(ing_section.replace(",", " ").replace("\n", " ").split())
    answer_tokens = {t for t in answer_tokens if len(t) > 3}  # skip short stopwords

    if not answer_tokens:
        return 1.0

    # Build reference token set from retrieved docs + available ingredients
    reference_text = " ".join(retrieved_docs + available_ingredients).lower()
    reference_tokens = set(reference_text.replace(",", " ").replace("\n", " ").split())

    grounded = answer_tokens & reference_tokens
    score = len(grounded) / len(answer_tokens)
    return round(score, 4)


def compute_answer_relevance(query: str, answer: str, embedder) -> float:
    """
    Answer Relevance: cosine similarity between query embedding and answer embedding.
    Target: >= 0.75
    """
    vecs = embedder.encode([query, answer], normalize_embeddings=True)
    score = float(np.dot(vecs[0], vecs[1]))
    return round(score, 4)


def compute_contextual_precision(answer: str, retrieved_docs: list[str], overlap_threshold: int = 3) -> float:
    """
    Contextual Precision: fraction of retrieved docs that share >= overlap_threshold
    ingredient tokens with the answer.

    Target: >= 0.65
    """
    if not retrieved_docs:
        return 0.0

    answer_tokens = set(answer.lower().split())
    precise = 0

    for doc in retrieved_docs:
        doc_tokens = set(doc.lower().split())
        overlap = len(answer_tokens & doc_tokens)
        if overlap >= overlap_threshold:
            precise += 1

    score = precise / len(retrieved_docs)
    return round(score, 4)


def compute_contextual_recall(answer: str, available_ingredients: list[str]) -> float:
    """
    Contextual Recall: fraction of user's available ingredients that appear in the answer.
    Target: >= 0.65
    """
    if not available_ingredients:
        return 1.0  # no constraints to check

    answer_lower = answer.lower()
    used = sum(1 for ing in available_ingredients if ing.lower() in answer_lower)
    score = used / len(available_ingredients)
    return round(score, 4)


# ── Main evaluation runner ────────────────────────────────────

def run_evaluation(
    eval_rows: list[dict],
    embedder,
    model_name: str = "hybrid_rag"
) -> pd.DataFrame:
    """
    Run all 4 metrics over a list of eval rows.

    Each eval_row must have:
        - query (str)
        - answer (str)
        - contexts (list of retrieved doc strings)
        - available_ingredients (list of str, optional)

    Args:
        eval_rows: list of dicts as described above
        embedder: loaded SentenceTransformer
        model_name: label for this model (baseline_llm / naive_rag / hybrid_rag)

    Returns:
        DataFrame with one row per query and columns for each metric
    """
    results = []

    for i, row in enumerate(eval_rows):
        query       = row["query"]
        answer      = row["answer"]
        contexts    = row.get("contexts", [])
        avail_ings  = row.get("available_ingredients", [])

        faithfulness        = compute_faithfulness(answer, contexts, avail_ings)
        answer_relevance    = compute_answer_relevance(query, answer, embedder)
        contextual_precision = compute_contextual_precision(answer, contexts)
        contextual_recall   = compute_contextual_recall(answer, avail_ings)

        results.append({
            "model":                 model_name,
            "query":                 query,
            "answer":                answer[:200],  # truncate for readability
            "faithfulness":          faithfulness,
            "answer_relevance":      answer_relevance,
            "contextual_precision":  contextual_precision,
            "contextual_recall":     contextual_recall,
        })

        print(
            f"[{i+1}/{len(eval_rows)}] F={faithfulness:.2f} "
            f"AR={answer_relevance:.2f} "
            f"CP={contextual_precision:.2f} "
            f"CR={contextual_recall:.2f} | {query[:40]}..."
        )

    df_results = pd.DataFrame(results)

    # Print summary
    print("\n── Summary ──────────────────────────────────")
    print(f"Model: {model_name}")
    print(f"  Faithfulness:          {df_results['faithfulness'].mean():.3f}  (target ≥ 0.70)")
    print(f"  Answer Relevance:      {df_results['answer_relevance'].mean():.3f}  (target ≥ 0.75)")
    print(f"  Contextual Precision:  {df_results['contextual_precision'].mean():.3f}  (target ≥ 0.65)")
    print(f"  Contextual Recall:     {df_results['contextual_recall'].mean():.3f}  (target ≥ 0.65)")

    return df_results


def save_results(df_results: pd.DataFrame, path: str = EVAL_RESULTS_PATH):
    """Append results to the eval CSV (creates it if it doesn't exist)."""
    if pd.io.common.file_exists(path):
        existing = pd.read_csv(path)
        combined = pd.concat([existing, df_results], ignore_index=True)
    else:
        combined = df_results

    combined.to_csv(path, index=False)
    print(f"Results saved to {path}")
