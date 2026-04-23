# app/gradio_app.py
# desktop layout
# All path fixes 

import os
import sys
import re
import ast
import pickle
import importlib.util
import numpy as np
import gradio as gr

# ── Path setup ─────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BASE_DIR        = os.environ.get("RECIPE_BASE_DIR", "/kaggle/working")
CHROMA_PATH     = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "recipes"

from generation.llm import generate_recipe, baseline_generate, naive_rag_generate
import pandas as pd
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_p3 = _load_module("pipeline", os.path.join(ROOT, "pipeline.py"))
person3_pipeline = _p3.person3_pipeline

print("Loading DataFrame...")
df = pd.read_parquet(os.path.join(BASE_DIR, "recipes_clean.parquet"))

def safe_parse(val):
    if isinstance(val, list): return val
    try: return ast.literal_eval(val)
    except: return []

df["ingredients"] = df["ingredients"].apply(safe_parse)
df["directions"]  = df["directions"].apply(safe_parse)

if "document" not in df.columns:
    df["document"] = df.apply(
        lambda r: f"{r['title']}. Ingredients: {', '.join(r['ingredients'])}", axis=1)
if "full_text" not in df.columns:
    df["full_text"] = df.apply(
        lambda r: f"# {r['title']}\n\n## Ingredients\n"
        + "\n".join(f"- {i}" for i in r["ingredients"])
        + "\n\n## Instructions\n"
        + "\n".join(f"{idx+1}. {s}" for idx, s in enumerate(r["directions"])), axis=1)

print("Loading BM25...")
with open(os.path.join(BASE_DIR, "bm25_index.pkl"), "rb") as f:
    bm25 = pickle.load(f)

print("Loading ChromaDB...")
client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)

print("Loading embedder...")
import torch
device   = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
print(f"All loaded. Embedder on {device}.")

# ── Retrieval ─────────────────────────────────────────────────
def tokenize(text):
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()

def retrieve(query, top_k=5):
    bm25_top  = np.argsort(bm25.get_scores(tokenize(query)))[::-1][:50].tolist()
    q_emb     = embedder.encode([query], normalize_embeddings=True).tolist()
    dense_r   = collection.query(query_embeddings=q_emb, n_results=50, include=["metadatas"])
    dense_top = [int(rid.split("_")[1]) for rid in dense_r["ids"][0]]
    scores    = {}
    for ranked in [bm25_top, dense_top]:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0) + 1.0 / (60 + rank + 1)
    fused = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
    return [{"title": df.iloc[i]["title"], "ingredients": df.iloc[i]["ingredients"],
             "full_text": df.iloc[i]["full_text"], "df_index": int(i)} for i in fused]

# ── Core pipeline ─────────────────────────────────────────────
def run_pipeline(query, ingredients_text, diet, appliance, time_limit, budget, servings, mode):
    if not query.strip():
        return "", "", "", "", ""

    available   = [i.strip() for i in ingredients_text.split(",") if i.strip()] if ingredients_text.strip() else []
    constraints = {}
    if available:            constraints["ingredients"] = available
    if diet != "None":       constraints["diet"]        = diet
    if appliance != "None":  constraints["appliance"]   = appliance
    if time_limit != "None": constraints["time"]        = time_limit
    if budget != "None":     constraints["budget"]      = budget

    retrieved  = retrieve(query)
    ref_titles = "\n".join(f"{i+1}. {r['title']}" for i, r in enumerate(retrieved))

    if mode == "Baseline (no RAG)":
        recipe = baseline_generate(query, constraints)
    elif mode == "Naive RAG":
        recipe = naive_rag_generate(query, retrieved, constraints)
    else:
        recipe = generate_recipe(query, retrieved, constraints=constraints)

    lines = recipe.split("\n")
    gen_ingredients, gen_steps = [], []
    in_ing, in_steps = False, False
    for line in lines:
        line = line.strip()
        if "ingredients:" in line.lower():                                    in_ing, in_steps = True, False;  continue
        if "instructions:" in line.lower() or "directions:" in line.lower(): in_steps, in_ing = True, False;  continue
        if "estimated time:" in line.lower() or "serves:" in line.lower():   in_ing, in_steps = False, False
        if in_ing and (line.startswith("-") or line.startswith("*")):
            gen_ingredients.append(line.lstrip("-* ").strip())
        if in_steps and re.match(r"^\d+\.", line):
            gen_steps.append(re.sub(r"^\d+\.\s*", "", line).strip())

    if not gen_ingredients: gen_ingredients = available or ["unknown"]
    if not gen_steps:       gen_steps       = [recipe[:200]]

    est_time, max_time = None, None
    t_match = re.search(r"estimated time[:\s]+(\d+)", recipe.lower())
    if t_match: est_time = int(t_match.group(1))
    if time_limit != "None":
        tm = re.search(r"(\d+)", time_limit)
        if tm: max_time = int(tm.group(1))

    try:
        p3 = person3_pipeline(
            recipe_steps=gen_steps,
            recipe_ingredients=gen_ingredients,
            available_ingredients=available if available else gen_ingredients,
            dietary_restrictions=[diet.lower()] if diet != "None" else None,
            available_appliances=[appliance.lower()] if appliance != "None" else None,
            max_time_minutes=max_time,
            estimated_time_minutes=est_time,
            servings=int(servings),
            top_k_substitutes=3
        )

        subs_out = ""
        if p3.get("missing_ingredients"):
            subs_out += f"**Missing:** {', '.join(p3['missing_ingredients'])}\n\n"
            for ing, subs in p3["substitutions"].items():
                subs_out += f"**{ing}** can be replaced with:\n"
                for s in subs:
                    flag = "IN" if s.get("is_indian") else "—"
                    subs_out += f"  [{flag}] {s['ingredient']}  (score: {s['final_score']})\n"
        else:
            subs_out = "All ingredients available — no substitutions needed."

        r = p3["reward"]
        reward_out = (
            f"{p3['verdict']}\n\n"
            f"| Metric | Score |\n|---|---|\n"
            f"| Coherence (30%) | {r['coherence_score']} |\n"
            f"| Constraint satisfaction (40%) | {r['constraint_satisfaction_score']} |\n"
            f"| Ingredient feasibility (30%) | {r['ingredient_feasibility_score']} |\n"
            f"| **Final reward** | **{r['final_reward']}** |"
        )

        c = p3["cost"]
        cost_out = (
            f"{c['budget_category']}\n\n"
            f"**Total:** {c['total_cost']}  ·  **Per serving:** {c['cost_per_serving']}\n\n"
            + "\n".join(f"- {item['ingredient']}: {item['cost']}"
                        for item in c["breakdown"] if item["cost"] != "unknown")
        )

    except Exception as e:
        subs_out   = f"Substitution unavailable ({str(e)[:80]})"
        reward_out = "Score unavailable"
        cost_out   = "Cost unavailable"

    return recipe, f"**References used:**\n{ref_titles}", subs_out, reward_out, cost_out


def run_image_pipeline(image, ingredients_text, diet, appliance, time_limit, budget, servings, mode):
    if image is None:
        return "No image uploaded.", "", "", "", "", ""
    try:
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image as PILImage
        DISH_LABELS = [
            "dal makhani", "butter chicken", "biryani", "paneer tikka",
            "aloo gobi", "rajma", "palak paneer", "dosa", "idli sambar",
            "fried rice", "pasta", "pizza", "burger", "salad",
            "soup", "noodles", "grilled chicken", "fish curry",
            "omelette", "chocolate cake", "aloo paratha"
        ]
        clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        pil_img        = PILImage.fromarray(image).convert("RGB")
        inputs         = clip_processor(
            text=[f"a photo of {l}" for l in DISH_LABELS],
            images=pil_img, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            probs = clip_model(**inputs).logits_per_image.softmax(dim=1)
        best_idx     = probs[0].argmax().item()
        detected     = DISH_LABELS[best_idx]
        conf         = round(probs[0][best_idx].item() * 100, 1)
        query        = f"{detected} recipe"
        detected_str = f"{detected}  ({conf}% confidence)"
    except Exception as e:
        detected_str = f"Could not identify: {str(e)[:60]}"
        query        = "recipe"

    recipe, refs, subs, reward, cost = run_pipeline(
        query, ingredients_text, diet, appliance, time_limit, budget, servings, mode)
    return detected_str, recipe, refs, subs, reward, cost


def run_constraint_pipeline(cuisine, diet, appliance, time_limit, budget, ingredients_text, servings, mode):
    parts = []
    if cuisine != "Any":     parts.append(f"{cuisine} cuisine")
    if diet != "None":       parts.append(f"{diet.lower()}")
    if time_limit != "None": parts.append(f"ready in {time_limit}")
    query = "Give me a recipe " + " and ".join(parts) if parts else "Give me a good recipe"
    if ingredients_text.strip(): query += f" using: {ingredients_text}"
    return run_pipeline(query, ingredients_text, diet, appliance, time_limit, budget, servings, mode)


# ── Options ───────────────────────────────────────────────────
DIET_OPTS      = ["None", "Vegetarian", "Vegan", "Jain", "Gluten Free", "Diabetic"]
APPLIANCE_OPTS = ["None", "Stovetop", "Pressure Cooker", "Microwave", "Oven", "Air Fryer"]
TIME_OPTS      = ["None", "15 minutes", "30 minutes", "45 minutes", "60 minutes", "90 minutes"]
BUDGET_OPTS    = ["None", "Under ₹50", "Under ₹150", "Under ₹300", "No limit"]
CUISINE_OPTS   = ["Any", "Indian", "South Indian", "Mughlai", "Italian", "Chinese", "Mexican", "Continental"]
MODE_OPTS      = ["Advanced RAG (RRF)", "Naive RAG", "Baseline (no RAG)"]

# ── CSS ───────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500&family=Inter:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #F5EDD8 !important;
    font-family: 'Inter', sans-serif !important;
    color: #2C1A0E !important;
    max-width: 1280px !important;
    margin: 0 auto !important;
}

h1, h2, h3, h4, p, span, div, label, textarea, input, select, button {
    color: #000000 !important;
}

.app-title {
    font-family: 'Lora', serif !important;
    font-size: 1.8rem !important;
    font-weight: 500 !important;
    color: #000000 !important;
    padding: 1.4rem 0 1rem;
    border-bottom: 1.5px solid #C9B07A;
    margin-bottom: 1.6rem;
    letter-spacing: -0.2px;
}

label span {
    font-size: 10px !important;
    font-weight: 500 !important;
    color: #7A5C2E !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
}

textarea, input[type=text], input[type=number] {
    background: #FDF6E3 !important;
    border: 1.5px solid #C9B07A !important;
    border-radius: 8px !important;
    color: #2C1A0E !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    padding: 10px 13px !important;
}
textarea:focus, input:focus {
    border-color: #8B6914 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(139,105,20,0.12) !important;
}
textarea::placeholder, input::placeholder {
    color: #A8896A !important;
}

select {
    background: #FDF6E3 !important;
    border: 1.5px solid #C9B07A !important;
    border-radius: 8px !important;
    color: #2C1A0E !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    padding: 8px 12px !important;
}

button.primary {
    background: #A8E6A3 !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 11px 30px !important;
    letter-spacing: 0.3px !important;
    transition: background 0.2s !important;
}
button.primary:hover {
    background: #94D89B !important;
}

button:not(.primary) {
    background: #C9F2C7 !important;
    color: #000000 !important;
    border: 1px solid #C9B07A !important;
    border-radius: 7px !important;
    font-size: 12px !important;
}

.block, .output-markdown, .prose {
    background: #FAF0D0 !important;
    border: 1.5px solid #D4BC82 !important;
    border-radius: 10px !important;
    color: #2C1A0E !important;
}

.tab-nav {
    border-bottom: 1.5px solid #C9B07A !important;
    margin-bottom: 1.5rem !important;
    background: transparent !important;
}
.tab-nav button {
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #7A5C2E !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2.5px solid transparent !important;
    padding: 9px 18px !important;
    margin-bottom: -1.5px !important;
    letter-spacing: 0.3px !important;
}
.tab-nav button.selected {
    color: #2C1A0E !important;
    border-bottom-color: #8B4513 !important;
}
.tab-nav button:hover {
    color: #4A2C0E !important;
}

input[type=range] { accent-color: #8B4513 !important; }
input[type=radio]  { accent-color: #8B4513 !important; }

.markdown-body, .prose p, .prose li, .prose h1, .prose h2, .prose h3 {
    color: #000000 !important;
}

table { border-collapse: collapse; width: 100%; }
th { background: #E8D5A3 !important; color: #000000 !important; padding: 7px 12px; font-size: 12px; font-weight: 500; }
td { color: #000000 !important; padding: 6px 12px; border-bottom: 1px solid #D4BC82; font-size: 12px; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #F5EDD8; }
::-webkit-scrollbar-thumb { background: #C9B07A; border-radius: 3px; }

footer { display: none !important; }

/* FORCE dropdown popup (the black box) to white */
div[role="listbox"] {
    background: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #C9B07A !important;
}

/* Each option inside dropdown */
div[role="option"] {
    background: #FFFFFF !important;
    color: #000000 !important;
}

/* Hover + selected state */
div[role="option"]:hover,
div[role="option"][aria-selected="true"] {
    background: #E8F5E9 !important;  /* soft pastel hover */
    color: #000000 !important;
}
"""

# ── UI ────────────────────────────────────────────────────────
with gr.Blocks(css=css, title="AI_Recipe") as demo:

    gr.HTML('<div class="app-title">AI_Recipe</div>')

    with gr.Tabs():

        # ── Tab 1: Kitchen ────────────────────────────────────
        with gr.Tab("Kitchen"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=360):
                    t1_query = gr.Textbox(
                        label="What do you want to cook?",
                        placeholder="Something warm with lentils and spices...",
                        lines=2
                    )
                    t1_ingredients = gr.Textbox(
                        label="Ingredients you have (comma separated)",
                        placeholder="tomato, onion, toor dal, cumin, ghee..."
                    )
                    with gr.Row():
                        t1_diet      = gr.Dropdown(DIET_OPTS,      label="Diet",      value="None")
                        t1_appliance = gr.Dropdown(APPLIANCE_OPTS, label="Appliance", value="None")
                    with gr.Row():
                        t1_time   = gr.Dropdown(TIME_OPTS,   label="Time limit", value="None")
                        t1_budget = gr.Dropdown(BUDGET_OPTS, label="Budget",     value="None")
                    t1_servings = gr.Slider(1, 8, value=2, step=1, label="Servings")
                    t1_mode     = gr.Radio(MODE_OPTS, value="Advanced RAG (RRF)", label="Model mode")
                    t1_btn      = gr.Button("Generate recipe →", variant="primary")

                with gr.Column(scale=2, min_width=520):
                    t1_recipe    = gr.Markdown(label="Recipe")
                    t1_retrieved = gr.Markdown(label="References used")
                    with gr.Row():
                        t1_subs = gr.Markdown(label="Substitutions")
                        t1_cost = gr.Markdown(label="Cost estimate")
            t1_reward_hidden = gr.Markdown(visible=False)

            t1_btn.click(
                fn=run_pipeline,
                inputs=[t1_query, t1_ingredients, t1_diet, t1_appliance,
                        t1_time, t1_budget, t1_servings, t1_mode],
                outputs=[t1_recipe, t1_retrieved, t1_subs, t1_reward_hidden, t1_cost]
            )

        # ── Tab 2: Photo ──────────────────────────────────────
        with gr.Tab("Photo"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=360):
                    t2_image       = gr.Image(type="numpy", label="Upload a dish photo")
                    t2_detected    = gr.Textbox(label="Detected dish", interactive=False)
                    t2_ingredients = gr.Textbox(
                        label="Ingredients you have (optional)",
                        placeholder="paneer, onion, tomato..."
                    )
                    with gr.Row():
                        t2_diet      = gr.Dropdown(DIET_OPTS,      label="Diet",      value="None")
                        t2_appliance = gr.Dropdown(APPLIANCE_OPTS, label="Appliance", value="None")
                    with gr.Row():
                        t2_time   = gr.Dropdown(TIME_OPTS,   label="Time limit", value="None")
                        t2_budget = gr.Dropdown(BUDGET_OPTS, label="Budget",     value="None")
                    t2_servings = gr.Slider(1, 8, value=2, step=1, label="Servings")
                    t2_mode     = gr.Radio(MODE_OPTS, value="Advanced RAG (RRF)", label="Model mode")
                    t2_btn      = gr.Button("Identify & generate →", variant="primary")

                with gr.Column(scale=2, min_width=520):
                    t2_recipe    = gr.Markdown(label="Recipe")
                    t2_retrieved = gr.Markdown(label="References used")
                    with gr.Row():
                        t2_subs = gr.Markdown(label="Substitutions")
                        t2_cost = gr.Markdown(label="Cost estimate")
            t2_reward_hidden = gr.Markdown(visible=False)

            t2_btn.click(
                fn=run_image_pipeline,
                inputs=[t2_image, t2_ingredients, t2_diet, t2_appliance,
                        t2_time, t2_budget, t2_servings, t2_mode],
                outputs=[t2_detected, t2_recipe, t2_retrieved, t2_subs,
                         t2_reward_hidden, t2_cost]
            )

        # ── Tab 3: Constraints ────────────────────────────────
        with gr.Tab("Constraints"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=360):
                    t3_cuisine    = gr.Dropdown(CUISINE_OPTS, label="Cuisine", value="Any")
                    with gr.Row():
                        t3_diet      = gr.Dropdown(DIET_OPTS,      label="Diet",      value="None")
                        t3_appliance = gr.Dropdown(APPLIANCE_OPTS, label="Appliance", value="None")
                    with gr.Row():
                        t3_time   = gr.Dropdown(TIME_OPTS,   label="Time limit", value="None")
                        t3_budget = gr.Dropdown(BUDGET_OPTS, label="Budget",     value="None")
                    t3_ingredients = gr.Textbox(
                        label="Ingredients you have",
                        placeholder="potato, onion, spices..."
                    )
                    t3_servings = gr.Slider(1, 8, value=2, step=1, label="Servings")
                    t3_mode     = gr.Radio(MODE_OPTS, value="Advanced RAG (RRF)", label="Model mode")
                    t3_btn      = gr.Button("Find recipe →", variant="primary")

                with gr.Column(scale=2, min_width=520):
                    t3_recipe    = gr.Markdown(label="Recipe")
                    t3_retrieved = gr.Markdown(label="References used")
                    with gr.Row():
                        t3_subs = gr.Markdown(label="Substitutions")
                        t3_cost = gr.Markdown(label="Cost estimate")
            t3_reward_hidden = gr.Markdown(visible=False)

            t3_btn.click(
                fn=run_constraint_pipeline,
                inputs=[t3_cuisine, t3_diet, t3_appliance, t3_time, t3_budget,
                        t3_ingredients, t3_servings, t3_mode],
                outputs=[t3_recipe, t3_retrieved, t3_subs, t3_reward_hidden, t3_cost]
            )

        # ── Tab 4: Metrics ────────────────────────────────────
        with gr.Tab("Metrics & Stats"):
            gr.Markdown("""
## Evaluation framework

### RAGAS automated metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Faithfulness | non-hallucinated ingredients / total ingredients listed | ≥ 0.70 |
| Answer relevance | cosine similarity (query embedding, answer embedding) | ≥ 0.75 |
| Contextual precision | retrieved docs with ≥3 ingredient overlap / total retrieved | ≥ 0.65 |
| Contextual recall | user ingredients used in answer / total user ingredients | ≥ 0.65 |

---

### MCTS reward score

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Coherence | 30% | Semantic flow between steps + stage order (prep → cook → serve) |
| Constraint satisfaction | 40% | Dietary rules, appliance compatibility, time budget |
| Ingredient feasibility | 30% | Available = 1.0 · Substitute available = 0.6 · Neither = 0.0 |

**Final reward** = 0.3 × Coherence + 0.4 × Constraints + 0.3 × Feasibility

Verdict thresholds: ≥ 0.75 Excellent · ≥ 0.55 Good · ≥ 0.35 Fair · < 0.35 Poor

---

### Model comparison

| Model | Faithfulness | Answer relevance | Ctx precision | Ctx recall |
|-------|-------------|-----------------|---------------|------------|
| Baseline LLM | Low | Medium | N/A | Low |
| Naive RAG | Medium | Medium | Medium | Medium |
| Advanced RAG (RRF) | **High** | **High** | **High** | **High** |

---

### Dataset & retrieval stats

| | |
|--|--|
| Dataset | RecipeNLG — first 50,000 rows |
| After filtering (≥ 3 ingredients) | 49,520 recipes |
| Embedding model | all-MiniLM-L6-v2 · 384 dimensions |
| Vector store | ChromaDB · HNSW index · cosine similarity · 341 MB |
| Sparse index | BM25Okapi via rank_bm25 · 12.6 MB |
| Fusion method | Reciprocal Rank Fusion · k=60 · pool size 50+50 |
| LLM | LLaMA 3.1 8B via Groq API |
| Cost estimator | Indian market prices (BigBasket / Blinkit averages) |
            """)

# ── Launch ────────────────────────────────────────────────────
if __name__ == "__main__":
    is_kaggle = os.path.exists("/kaggle")
    demo.launch(share=is_kaggle, server_name="0.0.0.0", server_port=7860)
