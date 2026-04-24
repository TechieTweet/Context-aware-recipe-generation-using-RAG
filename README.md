# Context-aware Recipe Generation using RAG
# RecipeRAG

A Retrieval-Augmented Generation system for personalised recipe recommendations with multimodal input, ingredient substitution, and cost estimation.

## Project Structure

```
recipeRAG/
├── config.py                  # All paths, model names, constants
├── requirements.txt
│
├── retrieval/
│   ├── embedder.py            # MiniLM embedding model loader
│   ├── vector_store.py        # ChromaDB build + query
│   ├── bm25_retriever.py      # BM25 sparse index build + query
│   └── hybrid_retriever.py    # RRF fusion — main retrieve function
│
├── generation/
│   └── llm.py                 # HF Inference API (Mistral-7B) — Person 2
│
├── evaluation/
│   └── ragas_eval.py          # Faithfulness, Answer Relevance, Precision, Recall
│
├── substitution/
│   └── substitutor.py         # FlavorDB + food2vec scorer — Person 3
│
├── app/
│   └── gradio_app.py          # Gradio demo UI
│
└── data/
    └── .gitkeep               # Actual data files stay on Kaggle (too large)
```

## Setup

```bash
pip install -r requirements.txt
```

## Running on Kaggle

1. Add the `recipe-nlg` dataset as input
2. Set accelerator to GPU T4
3. Run `pipeline.ipynb` top to bottom
4. For the demo, run `app/gradio_app.py`

## Team

- **Person 1(Shreya parashar)** — Data pipeline, ChromaDB, BM25, hybrid retrieval, RAGAS evaluation
- **Person 2(Swathi M)** — LLM generation, CLIP image input, Gradio UI
- **Person 3(Srujana T)** — Ingredient substitution, cost estimation, human evaluation rubric
