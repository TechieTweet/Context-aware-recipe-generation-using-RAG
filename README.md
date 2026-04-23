# Context-aware Recipe Generation using RAG
## RecipeRAG

A Retrieval-Augmented Generation system for personalized recipe recommendations with multimodal input, ingredient substitution, cost estimation, and human-in-the-loop evaluation.

---

## Project Structure

```
Context-aware-recipe-generation-using-RAG/
│
├── config.py                          # Central configuration: paths, models, constants
├── pipeline.py                        # Person 3 pipeline: master orchestrator
├── api.py                             # Flask API endpoints for all components
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
│
├── retrieval/                         # Person 1: Hybrid retrieval system
│   ├── __init__.py
│   ├── embedder.py                    # Loads SentenceTransformer MiniLM model
│   ├── bm25_retriever.py              # BM25 sparse retrieval (keyword-based)
│   ├── vector_store.py                # ChromaDB dense retrieval (semantic search)
│   └── hybrid_retriever.py            # Reciprocal Rank Fusion (RRF) merging
│
├── generation/                        # Person 2: LLM generation layer
│   ├── __init__.py
│   └── llm.py                         # HuggingFace Inference API (Mistral-7B)
│                                      # 3 modes: baseline, naive_rag, advanced_rag
│
├── evaluation/                        # Person 1: RAGAS evaluation metrics
│   ├── __init__.py
│   └── ragas_eval.py                  # Faithfulness, Answer Relevance, 
│                                      # Contextual Precision, Contextual Recall
│
├── substitution/                      # Person 3: Ingredient substitution
│   ├── __init__.py
│   ├── substitution.py                # Wrapper for substitution + cost + reward
│   └── substitution_model.py          # FlavorDB + food2vec embedding scorer
│
├── mcts/                              # Person 3: Recipe evaluation via rewards
│   ├── __init__.py
│   └── reward_function.py             # MCTS-based coherence, constraints, feasibility
│
├── cost/                              # Person 3: Cost estimation
│   └── cost_estimator.py              # INR cost breakdown by ingredient
│
├── app/                               # Person 2: User interfaces
│   ├── __init__.py
│   ├── gradio_app.py                  # Gradio web UI 
│   └── clip_classifier.py             # CLIP image → dish classification
│
└── data/                              # Data handling & exploration
    ├── explore_flavourdb.py           # FlavorDB analysis notebook
    ├── fetch_flavourdb.py             # FlavorDB download utility
    ├── flavourdb.json                 # FlavorDB ingredient database
    ├── indian_ingredients.py          # Indian ingredient definitions
    └── indian_substitutes_kb.py       # Indian substitution knowledge base
```

---

## Pipeline & Architecture

### 1️.**Data Input Layer (User Input)**
User provides via Gradio UI:
- **Text query**: "butter chicken" OR **Image**: dish photo (CLIP classification)
- **Available ingredients**: comma-separated list
- **Constraints**: dietary restrictions, appliances, time budget, cost budget, servings

```python
# Processed in app/gradio_app.py run_pipeline()
query, available_ingredients, constraints → [to retrieval]
```

---

### 2️.  **Retrieval Layer (Person 1: Hybrid Search)**
**Goal**: Find top-K relevant recipes from 50,000+ recipe database

#### Step 2-A: **BM25 Sparse Retrieval**
- Keyword-based ranking using BM25Okapi algorithm
- Fast, deterministic, great for exact ingredient matches
- Returns top 50 candidates

```python
# retrieval/bm25_retriever.py
query_bm25(bm25_index, query) → [recipe_idx_1, recipe_idx_2, ...]
```

#### Step 2-B: **Dense Vector Retrieval**
- Semantic similarity using SentenceTransformer MiniLM embeddings
- ChromaDB stores & searches 50,000 recipe embeddings
- Captures meaning beyond keywords
- Returns top 50 candidates

```python
# retrieval/vector_store.py
query_dense(collection, embedder, query) → [recipe_idx_5, recipe_idx_12, ...]
```

#### Step 2-C: **Reciprocal Rank Fusion (RRF)**
- Merges BM25 + dense rankings using RRF formula: `score = Σ(1 / (k + rank + 1))`
- Avoids over-reliance on either method
- **Final result**: Top 5 recipes ranked by combined score

```python
# retrieval/hybrid_retriever.py
hybrid_retrieve(query, df, bm25, collection, embedder) → [top_5_recipes]
```

---

### 3️. **Generation Layer (Person 2: LLM)**
**Goal**: Generate a personalized recipe given query + retrieved context

#### Three Generation Modes:

**Mode 1: Baseline (No RAG)**
- Direct LLM call with user query only
- No retrieval context
- Baseline for comparison

**Mode 2: Naive RAG**
- Simple retrieval + context concatenation
- No reranking or intelligent fusion
- Baseline for evaluation

**Mode 3: Advanced RAG (Recommended)**
- Uses hybrid retrieved recipes as context
- Builds rich prompt with constraints
- Instructs LLM to prioritize available ingredients
- LLM: Mistral-7B via HuggingFace Inference API

```python
# generation/llm.py → build_prompt() + generate_recipe()
generate_recipe(query, retrieved_recipes, constraints) → [recipe_text]
```

**Generated recipe structure:**
```
# Recipe Title
## Ingredients
- ingredient 1: quantity
- ingredient 2: quantity
...
## Instructions
1. Step 1
2. Step 2
...
Estimated time: X minutes
Serves: Y
```

---

### 4️. **Post-Processing Layer (Person 3: Enrichment)**
**Goal**: Validate, enhance, and score the generated recipe

#### Step 4-A: **Parse Generated Recipe**
Extract from generated recipe:
- `gen_ingredients`: list of ingredient strings
- `gen_steps`: list of cooking step strings
- `est_time_minutes`: parsed from "Estimated time: X"

#### Step 4-B: **Ingredient Substitution**
- Find missing ingredients by comparing `gen_ingredients` vs `available_ingredients`
- For each missing ingredient, query FlavorDB + food2vec embeddings
- Rank substitutes by similarity (embedding score) + Indian prioritization boost
- Return top 3 substitutes per ingredient

```python
# substitution/substitution_model.py + substitution.py
get_substitutes(ingredient, top_k=3) → [{"substitute": "X", "score": 0.85, "is_indian": true}, ...]
```

#### Step 4-C: **Recipe Reward Scoring (MCTS)**
Evaluate recipe feasibility across 3 dimensions:

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| **Coherence** | 30% | Are steps logical? Do ingredients flow naturally? |
| **Constraint Satisfaction** | 40% | Time, appliances, diet, budget constraints met? |
| **Ingredient Feasibility** | 30% | Can user execute with available + substitute ingredients? |

**Final Reward** = weighted average; ranges 0–1

```python
# mcts/reward_function.py
compute_reward(recipe_steps, ingredients, available_ingredients, constraints) → {
    "coherence_score": 0.75,
    "constraint_satisfaction_score": 0.80,
    "ingredient_feasibility_score": 0.70,
    "final_reward": 0.75
}
```

**Verdict Logic:**
-  **≥ 0.75**: Excellent — highly feasible
-  **0.55–0.75**: Good — feasible with minor adjustments
-  **0.35–0.55**: Fair — needs substitutions
-  **< 0.35**: Poor — not feasible under constraints

#### Step 4-D: **Cost Estimation**
- Break down recipe cost by ingredient (INR)
- Account for servings
- Compare against budget constraint

```python
# cost/cost_estimator.py
estimate_cost(ingredients, servings=2) → {
    "total_cost": "₹250",
    "cost_per_serving": "₹125",
    "budget_category": "Budget",
    "breakdown": [{"ingredient": "chicken", "cost": "₹120"}, ...]
}
```

#### Step 4-E: **Master Pipeline Orchestration**
All components unified in one call:

```python
# pipeline.py → person3_pipeline()
person3_pipeline(
    recipe_steps=[...],
    recipe_ingredients=[...],
    available_ingredients=[...],
    dietary_restrictions=[...],
    max_time_minutes=30,
    servings=2
) → {
    "missing_ingredients": [...],
    "substitutions": {...},
    "reward": {...},
    "cost": {...},
    "verdict": "Excellent"
}
```

---

### 5️. **Evaluation Layer (Person 1: RAGAS Metrics)**
**Goal**: Quantify generation quality across 4 metrics

#### Metric 1: **Faithfulness** (Target ≥ 0.70)
> *Is the recipe honest? Does it only use ingredients from retrieved docs or your available ingredients?*
- Extracts ingredient tokens from answer
- Checks coverage against reference docs + available ingredients
- Score = grounded_ingredients / total_ingredients

```python
# evaluation/ragas_eval.py → compute_faithfulness()
```

#### Metric 2: **Answer Relevance** (Target ≥ 0.75)
> *Does the recipe match what you asked for?*
- Embeds both query and recipe using MiniLM
- Computes cosine similarity between embeddings
- Score ranges 0–1; 1 = perfectly relevant

```python
# evaluation/ragas_eval.py → compute_answer_relevance()
score = cosine_similarity(embed(query), embed(answer))
```

#### Metric 3: **Contextual Precision** (Target ≥ 0.65)
> *Out of all recipes we retrieved, how many were actually useful?*
- Counts retrieved docs sharing ≥3 ingredient tokens with answer
- Score = relevant_docs / total_retrieved_docs

```python
# evaluation/ragas_eval.py → compute_contextual_precision()
```

#### Metric 4: **Contextual Recall** (Target ≥ 0.65)
> *Did we use all ingredients you said you have?*
- Checks what fraction of user's available ingredients appear in recipe
- Score = used_ingredients / available_ingredients

```python
# evaluation/ragas_eval.py → compute_contextual_recall()
```

---

### 6️. **User Interface Layer**

#### **Gradio Web App** (Person 2)
- Multi-tab interface for different generation modes
- Real-time generation + enrichment
- Displays retrieved recipes, substitutions, cost, reward scores

```bash
python app/gradio_app.py
```

#### **REST API** (Flask)
- `/substitute` — Ingredient substitution
- `/score` — Recipe reward scoring
- `/cost` — Cost estimation
- `/pipeline` — Full pipeline in one call

```bash
python api.py
```

#### **CLIP Image Classifier** (Person 2)
- Classifies uploaded dish images to dish names
- Converts to text query for retrieval
- ~30 common dish labels (dal makhani, butter chicken, biryani, etc.)

```python
# app/clip_classifier.py
classify_dish(image) → "butter chicken"
```

---

## Configuration

All paths, model names, and constants are centralized in `config.py`:

```python
# Paths
BASE_DIR = os.environ.get("RECIPE_BASE_DIR", "/kaggle/working")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
BM25_INDEX_PATH = os.path.join(BASE_DIR, "bm25_index.pkl")
DATAFRAME_PATH = os.path.join(BASE_DIR, "recipes_clean.parquet")

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION = "recipes"

# Retrieval
BM25_CANDIDATES = 50      # Pool for BM25 before RRF
DENSE_CANDIDATES = 50     # Pool for dense before RRF
RRF_K = 60                # RRF constant (standard from literature)
TOP_K = 5                 # Final results returned

# Evaluation
NUM_EVAL_QUERIES = 10
```

---

## Quick Start

### Installation

```bash
# Clone repo
git clone <repo-url>
cd Context-aware-recipe-generation-using-RAG

# Install dependencies
pip install -r requirements.txt
```

### On Kaggle

1. **Add input dataset**: Search & add `recipe-nlg` dataset
2. **Enable GPU**: Settings → Accelerator → GPU T4 (or better)
3. **Create notebook** or upload existing pipeline
4. **Run pipeline** to build indexes (BM25 + ChromaDB)
5. **Launch Gradio UI**:
   ```bash
   python app/gradio_app.py
   ```

### Locally

```bash
# Ensure you have recipe data or set RECIPE_BASE_DIR
export RECIPE_BASE_DIR=/path/to/data

# Build indexes (if first time)
python -c "from retrieval import build_bm25_index, build_vector_store; ..."

# Run Gradio UI
python app/gradio_app.py

# Or run Flask API
python api.py  # Runs on http://localhost:5000
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       USER INPUT                             │
│     Query (text or image) + Constraints (diet, time, etc.)   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  RETRIEVAL (Person 1)                        │
│  BM25 (50) + Dense (50) → RRF Fusion → Top 5 Recipes       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  GENERATION (Person 2)                       │
│           LLM (Mistral-7B) → Generated Recipe               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               POST-PROCESSING (Person 3)                     │
│  ├─ Substitution (FlavorDB)                                 │
│  ├─ Reward Scoring (MCTS)                                   │
│  └─ Cost Estimation (INR)                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  EVALUATION (Person 1)                       │
│  Faithfulness + Answer Relevance + Precision + Recall       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    USER OUTPUT                               │
│  Final Recipe + Substitutions + Cost + Reward + Metrics     │
└─────────────────────────────────────────────────────────────┘
```

---

##  Team & Responsibilities

| Person | Role | Components |
|--------|------|------------|
| **Person 1** | Data & Evaluation | Hybrid Retrieval (BM25 + Dense + RRF), RAGAS Evaluation (4 metrics), Dataset pipeline | UI
| **Person 2** | Generation & UI | LLM generation (Mistral-7B), Gradio web UI, CLIP image classifier, Flask API |
| **Person 3** | Enrichment & Scoring | Ingredient substitution (FlavorDB), MCTS reward function, cost estimation, human evaluation |

---

## 📈 Key Metrics & Thresholds

### Evaluation Targets
| Metric | Target | Status |
|--------|--------|--------|
| Faithfulness | ≥ 0.70 | Prevents hallucination |
| Answer Relevance | ≥ 0.75 | Ensures query match |
| Contextual Precision | ≥ 0.65 | Relevant retrieved docs |
| Contextual Recall | ≥ 0.65 | Uses available ingredients |

### Recipe Feasibility (Reward Score)
| Score | Verdict | Meaning |
|-------|---------|---------|
| ≥ 0.75 | Excellent | Highly feasible, well-suited |
| 0.55–0.75 | Good | Feasible with adjustments |
| 0.35–0.55 | Fair | Needs substitutions |
| < 0.35 | Poor | Not feasible |

---

## Additional Files
- **requirements.txt** — Python package dependencies

---

## Troubleshooting

**Issue: ChromaDB not loading?**
- Ensure you built the vector store first: `python build_vector_store.py`
- Check `CHROMA_DB_PATH` in `config.py`

**Issue: BM25 index too large?**
- Reduce dataset size in `config.py` → `NUM_RECIPES`

**Issue: LLM API errors?**
- Verify HuggingFace API key is set
- Check internet connection

**Issue: Out of memory?**
- Reduce `EMBED_BATCH_SIZE` in `config.py` (default 1024 for GPU, 64 for CPU)

---

## References

- **RAGAS**: [RAG Assessment Suite](https://docs.ragas.io/)
- **MiniLM**: [SentenceTransformer MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Mistral-7B**: [Mistral Model](https://mistral.ai/)

