# config.py — central config for all paths, model names, and constants
# Import this in every other file instead of hardcoding paths

import os

# ── Paths ─────────────────────────────────────────────────────
# On Kaggle, working directory is /kaggle/working
# Locally, defaults to ./data/
BASE_DIR = os.environ.get("RECIPE_BASE_DIR", "/kaggle/working")

CHROMA_DB_PATH    = os.path.join(BASE_DIR, "chroma_db")
BM25_INDEX_PATH   = os.path.join(BASE_DIR, "bm25_index.pkl")
DATAFRAME_PATH    = os.path.join(BASE_DIR, "recipes_clean.parquet")
EVAL_RESULTS_PATH = os.path.join(BASE_DIR, "eval_results.csv")

# ── Dataset ───────────────────────────────────────────────────
CSV_PATH = "/kaggle/input/datasets/shreyaparashar101/recipe-nlg/dataset/full_dataset.csv"
NUM_RECIPES = 50_000
MIN_INGREDIENTS = 3

# ── Models ────────────────────────────────────────────────────
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION = "recipes"

# ── Retrieval ─────────────────────────────────────────────────
EMBED_BATCH_SIZE    = 1024   # for T4 GPU; reduce to 64 for CPU
BM25_CANDIDATES     = 50     # how many BM25 results to feed into RRF
DENSE_CANDIDATES    = 50     # how many dense results to feed into RRF
RRF_K               = 60     # standard RRF constant
TOP_K               = 5      # final number of results returned

# ── Evaluation ────────────────────────────────────────────────
NUM_EVAL_QUERIES = 10
