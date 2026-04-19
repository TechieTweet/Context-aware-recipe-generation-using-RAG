# retrieval/vector_store.py
# Handles building and querying the ChromaDB vector store

import chromadb
import pandas as pd
import time
from config import CHROMA_DB_PATH, CHROMA_COLLECTION, EMBED_BATCH_SIZE


def build_vector_store(df: pd.DataFrame, embedder) -> chromadb.Collection:
    """
    Embed all recipes and store them in ChromaDB.
    Deletes existing collection if present (clean rebuild).

    Args:
        df: cleaned recipes DataFrame with columns: title, ingredients, directions, document, full_text
        embedder: loaded SentenceTransformer model

    Returns:
        ChromaDB collection
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Clean rebuild
    try:
        client.delete_collection(CHROMA_COLLECTION)
        print("Deleted existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    documents  = df["document"].tolist()
    ids        = [f"recipe_{i}" for i in range(len(df))]
    metadatas  = [
        {
            "title":      str(row["title"]),
            "ingredients": "||".join(row["ingredients"]),
            "full_text":  row["full_text"][:2000]
        }
        for _, row in df.iterrows()
    ]

    total_batches = (len(documents) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    start = time.time()

    for i in range(total_batches):
        s, e = i * EMBED_BATCH_SIZE, min((i + 1) * EMBED_BATCH_SIZE, len(documents))

        batch_embeddings = embedder.encode(
            documents[s:e],
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True
        ).tolist()

        collection.add(
            ids=ids[s:e],
            embeddings=batch_embeddings,
            documents=documents[s:e],
            metadatas=metadatas[s:e]
        )

        pct = (i + 1) / total_batches * 100
        print(f"[{pct:.1f}%] Batch {i+1}/{total_batches} — {e}/{len(documents)} recipes — {time.time()-start:.0f}s")

    print(f"\nDone! {collection.count()} documents in ChromaDB. Time: {(time.time()-start)/60:.1f} min")
    return collection


def load_vector_store() -> chromadb.Collection:
    """
    Load an existing ChromaDB collection from disk.
    Call this instead of build_vector_store() after the first run.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(CHROMA_COLLECTION)
    print(f"ChromaDB loaded: {collection.count()} documents")
    return collection


def query_dense(collection: chromadb.Collection, embedder, query: str, n_results: int = 50):
    """
    Run a dense vector search against ChromaDB.

    Returns:
        List of integer DataFrame indices of the top results
    """
    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["metadatas", "documents", "distances"]
    )

    ids = results["ids"][0]  # e.g. ['recipe_42', 'recipe_7', ...]
    return [int(rid.split("_")[1]) for rid in ids]
