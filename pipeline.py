# pipeline.py
# Run this ONCE to build the vector store, BM25 index, and save the parquet file.
# After this, just run app/gradio_app.py directly.

import os
import pandas as pd
from config import (
    CSV_PATH, DATAFRAME_PATH, NUM_RECIPES, MIN_INGREDIENTS
)
from retrieval.embedder import load_embedder
from retrieval.vector_store import build_vector_store
from retrieval.bm25_retriever import build_bm25_index


# ── Step 1: Load and clean the dataset ─────────────────────────
print("=" * 50)
print("STEP 1: Loading and cleaning RecipeNLG dataset...")
print("=" * 50)

df = pd.read_csv(CSV_PATH, nrows=NUM_RECIPES)
print(f"Loaded {len(df)} rows from CSV.")

# Keep only needed columns
df = df[["title", "ingredients", "directions"]].copy()

# Drop nulls
df.dropna(subset=["title", "ingredients", "directions"], inplace=True)

# Parse ingredients — stored as string representation of a list in the CSV
import ast

def parse_list_column(val):
    try:
        parsed = ast.literal_eval(val)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []

df["ingredients"] = df["ingredients"].apply(parse_list_column)
df["directions"]  = df["directions"].apply(parse_list_column)

# Filter out recipes with too few ingredients
df = df[df["ingredients"].apply(len) >= MIN_INGREDIENTS].reset_index(drop=True)
print(f"After filtering (min {MIN_INGREDIENTS} ingredients): {len(df)} recipes.")

# Build document string — what gets embedded and searched
# Format: "title. ingredient1, ingredient2. step1 step2..."
df["document"] = df.apply(
    lambda row: (
        row["title"] + ". " +
        ", ".join(row["ingredients"]) + ". " +
        " ".join(row["directions"])[:300]
    ),
    axis=1
)

# Build full_text — used as context in the LLM prompt
df["full_text"] = df.apply(
    lambda row: (
        f"Title: {row['title']}\n"
        f"Ingredients: {', '.join(row['ingredients'])}\n"
        f"Instructions: {' '.join(row['directions'])}"
    ),
    axis=1
)

# ── Step 2: Save cleaned dataframe ─────────────────────────────
print("\n" + "=" * 50)
print("STEP 2: Saving cleaned dataframe...")
print("=" * 50)

os.makedirs(os.path.dirname(DATAFRAME_PATH), exist_ok=True)
df.to_parquet(DATAFRAME_PATH, index=False)
print(f"Saved to {DATAFRAME_PATH}")


# ── Step 3: Load embedder ───────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 3: Loading embedding model...")
print("=" * 50)

embedder = load_embedder()


# ── Step 4: Build ChromaDB vector store ────────────────────────
print("\n" + "=" * 50)
print("STEP 4: Building ChromaDB vector store...")
print("         This will take 10-20 mins on CPU for 50K recipes.")
print("=" * 50)

collection = build_vector_store(df, embedder)


# ── Step 5: Build BM25 index ───────────────────────────────────
print("\n" + "=" * 50)
print("STEP 5: Building BM25 index...")
print("=" * 50)

bm25 = build_bm25_index(df)


# ── Done ────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("ALL DONE. Pipeline complete.")
print(f"  Recipes indexed : {len(df)}")
print(f"  Parquet saved   : {DATAFRAME_PATH}")
print(f"  ChromaDB saved  : in data/chroma_db/")
print(f"  BM25 saved      : in data/bm25_index.pkl")
print("\nYou can now run the app:")
print("  python app/gradio_app.py")
print("=" * 50)