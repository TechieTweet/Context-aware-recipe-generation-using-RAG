# config.py
import os

# ── Base directory — everything stored relative to project root ──
BASE_DIR = os.environ.get("RECIPE_BASE_DIR", os.path.join(os.path.dirname(__file__), "data"))

CHROMA_DB_PATH    = os.path.join(BASE_DIR, "chroma_db")
BM25_INDEX_PATH   = os.path.join(BASE_DIR, "bm25_index.pkl")
DATAFRAME_PATH    = os.path.join(BASE_DIR, "recipes_clean.parquet")
EVAL_RESULTS_PATH = os.path.join(BASE_DIR, "eval_results.csv")

# ── Dataset — download from Kaggle and point this to the CSV ──
CSV_PATH = os.path.join(BASE_DIR, "full_dataset.csv")
NUM_RECIPES = 50_000
MIN_INGREDIENTS = 3

# ── Models ────────────────────────────────────────────────────
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION = "recipes"

# ── Retrieval ─────────────────────────────────────────────────
EMBED_BATCH_SIZE    = 256    # reduced from 1024 — that was for Kaggle T4 GPU
BM25_CANDIDATES     = 50
DENSE_CANDIDATES    = 50
RRF_K               = 60
TOP_K               = 5

# ── Evaluation ────────────────────────────────────────────────
NUM_EVAL_QUERIES = 10