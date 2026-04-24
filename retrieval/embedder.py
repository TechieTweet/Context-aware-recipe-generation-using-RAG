# retrieval/embedder.py
# Handles loading the MiniLM model and embedding text

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL
import torch


def load_embedder():
    """
    Load MiniLM embedding model onto GPU if available, else CPU.
    Returns the SentenceTransformer model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model '{EMBEDDING_MODEL}' on {device}...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    print(f"Embedder ready on {model.device}")
    return model


def embed_texts(model, texts: list[str], normalize: bool = True) -> list[list[float]]:
    """
    Embed a list of strings using the given SentenceTransformer model.

    Args:
        model: loaded SentenceTransformer
        texts: list of strings to embed
        normalize: whether to L2-normalize embeddings (required for cosine similarity)

    Returns:
        List of embedding vectors as Python lists
    """
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=normalize
    )
    return embeddings.tolist()
