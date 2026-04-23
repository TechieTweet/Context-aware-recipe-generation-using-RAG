# app/gradio_app.py
# Cafe-vibes UI — warm, clean, minimal
# All path fixes baked in — works on Kaggle and locally

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

# ── Data + index paths (Kaggle vs local) ──────────────────────
BASE_DIR        = os.environ.get("RECIPE_BASE_DIR", "/kaggle/working")
CHROMA_PATH     = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "recipes"

# ── Imports from repo ─────────────────────────────────────────
from generation.llm import generate_recipe, baseline_generate, naive_rag_generate
import pandas as pd
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ── Load Person 3 pipeline from repo root ─────────────────────
def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_p3 = _load_module("pipeline", os.path.join(ROOT, "pipeline.py"))
person3_pipeline = _p3.person3_pipeline

# ── Load all components ────────────────────────────────────────
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
        lambda r: f"{r['title']}. Ingredients: {', '.join(r['ingredients'])}", axis=1
    )
if "full_text" not in df.columns:
    df["full_text"] = df.apply(
        lambda r: f"# {r['title']}\n\n## Ingredients\n"
        + "\n".join(f"- {i}" for i in r["ingredients"])
        + "\n\n## Instructions\n"
        + "\n".join(f"{idx+1}. {s}" for idx, s in enumerate(r["directions"])),
        axis=1
    )

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
    return [
        {
            "title":       df.iloc[i]["title"],
            "ingredients": df.iloc[i]["ingredients"],
            "full_text":   df.iloc[i]["full_text"],
            "df_index":    int(i)
        }
        for i in fused
    ]


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
    ref_titles = "\n".join(f"  {i+1}. {r['title']}" for i, r in enumerate(retrieved))

    # Generation mode
    if mode == "Baseline (no RAG)":
        recipe = baseline_generate(query, constraints)
    elif mode == "Naive RAG":
        recipe = naive_rag_generate(query, retrieved, constraints)
    else:
        recipe = generate_recipe(query, retrieved, constraints=constraints)

    # Parse generated recipe for Person 3
    lines = recipe.split("\n")
    gen_ingredients, gen_steps = [], []
    in_ing, in_steps = False, False
    for line in lines:
        line = line.strip()
        if "ingredients:" in line.lower():   in_ing, in_steps = True, False;  continue
        if "instructions:" in line.lower() or "directions:" in line.lower(): in_steps, in_ing = True, False; continue
        if "estimated time:" in line.lower() or "serves:" in line.lower():   in_ing, in_steps = False, False
        if in_ing and (line.startswith("-") or line.startswith("*")):
            gen_ingredients.append(line.lstrip("-* ").strip())
        if in_steps and re.match(r"^\d+\.", line):
            gen_steps.append(re.sub(r"^\d+\.\s*", "", line).strip())

    if not gen_ingredients: gen_ingredients = available or ["unknown"]
    if not gen_steps:       gen_steps       = [recipe[:200]]

    est_time = None
    t_match  = re.search(r"estimated time[:\s]+(\d+)", recipe.lower())
    if t_match: est_time = int(t_match.group(1))

    max_time = None
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
                subs_out += f"**{ing}** →\n"
                for s in subs:
                    flag = "🇮🇳" if s.get("is_indian") else "🌍"
                    subs_out += f"  {flag} {s['ingredient']}  _{s['final_score']}_\n"
        else:
            subs_out = "✅ All ingredients available — no substitutions needed."

        r = p3["reward"]
        reward_out = (
            f"{p3['verdict']}\n\n"
            f"| Component | Score |\n|---|---|\n"
            f"| Coherence (30%) | {r['coherence_score']} |\n"
            f"| Constraint satisfaction (40%) | {r['constraint_satisfaction_score']} |\n"
            f"| Ingredient feasibility (30%) | {r['ingredient_feasibility_score']} |\n"
            f"| **Final reward** | **{r['final_reward']}** |"
        )

        c        = p3["cost"]
        cost_out = (
            f"{c['budget_category']}\n\n"
            f"**Total:** {c['total_cost']}  ·  **Per serving:** {c['cost_per_serving']}\n\n"
            + "\n".join(f"- {item['ingredient']}: {item['cost']}" for item in c["breakdown"] if item["cost"] != "unknown")
        )

    except Exception as e:
        subs_out   = f"Unavailable ({str(e)[:100]})"
        reward_out = "Unavailable"
        cost_out   = "Unavailable"

    return recipe, f"**References:**\n{ref_titles}", subs_out, reward_out, cost_out


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
            images=pil_img, return_tensors="pt", padding=True
        ).to(device)
        import torch
        with torch.no_grad():
            probs = clip_model(**inputs).logits_per_image.softmax(dim=1)
        best_idx     = probs[0].argmax().item()
        detected     = DISH_LABELS[best_idx]
        conf         = round(probs[0][best_idx].item() * 100, 1)
        query        = f"{detected} recipe"
        detected_str = f"{detected}  ({conf}% confidence)"
    except Exception as e:
        detected_str = f"Could not identify dish: {str(e)[:60]}"
        query        = "recipe"

    recipe, refs, subs, reward, cost = run_pipeline(
        query, ingredients_text, diet, appliance, time_limit, budget, servings, mode
    )
    return detected_str, recipe, refs, subs, reward, cost


def run_constraint_pipeline(cuisine, diet, appliance, time_limit, budget, ingredients_text, servings, mode):
    parts  = []
    if cuisine != "Any":        parts.append(f"{cuisine} cuisine")
    if diet != "None":          parts.append(f"{diet.lower()}")
    if time_limit != "None":    parts.append(f"ready in {time_limit}")
    query  = "Give me a recipe " + " and ".join(parts) if parts else "Give me a good recipe"
    if ingredients_text.strip(): query += f" using: {ingredients_text}"
    return run_pipeline(query, ingredients_text, diet, appliance, time_limit, budget, servings, mode)


# ── CSS ───────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;1,400&family=Jost:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #FEF9ED !important;
    font-family: 'Jost', sans-serif !important;
    color: #3D2B0E !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 2.5rem 2rem !important;
}

.app-header {
    text-align: center;
    padding: 1.8rem 0 1.5rem;
    border-bottom: 1px solid #E0CFA0;
    margin-bottom: 2rem;
}

.app-header h1 {
    font-family: 'Lora', serif !important;
    font-size: 2.2rem !important;
    font-weight: 500 !important;
    color: #3D2B0E !important;
    margin: 0 0 0.3rem !important;
    letter-spacing: -0.3px;
}

.app-header p {
    color: #9C845A !important;
    font-size: 13px !important;
    font-weight: 300 !important;
    margin: 0 !important;
    letter-spacing: 0.4px;
}

.tab-nav { border-bottom: 1px solid #E0CFA0 !important; margin-bottom: 1.5rem; }
.tab-nav button {
    font-family: 'Jost', sans-serif !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    color: #9C845A !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 18px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    margin-bottom: -1px !important;
}
.tab-nav button.selected {
    color: #3D2B0E !important;
    border-bottom-color: #C9A84C !important;
}

label span {
    font-family: 'Jost', sans-serif !important;
    font-size: 10px !important;
    font-weight: 500 !important;
    color: #9C845A !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
}

textarea, input[type="text"], input[type="number"] {
    background: #FEFCF4 !important;
    border: 1px solid #E0CFA0 !important;
    border-radius: 8px !important;
    color: #3D2B0E !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
}

textarea:focus, input:focus {
    border-color: #C9A84C !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(201,168,76,0.12) !important;
}

select {
    background: #FEFCF4 !important;
    border: 1px solid #E0CFA0 !important;
    border-radius: 8px !important;
    color: #3D2B0E !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 12px !important;
    padding: 8px 12px !important;
}

button.primary {
    background: #3D2B0E !important;
    color: #F5E6B2 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 12px 32px !important;
    letter-spacing: 0.5px !important;
}
button.primary:hover { background: #7A5C2E !important; }

.output-markdown, .block {
    background: #F7F0DA !important;
    border: 1px solid #EDE3C3 !important;
    border-radius: 10px !important;
}

.prose {
    font-family: 'Jost', sans-serif !important;
    font-size: 13px !important;
    line-height: 1.85 !important;
    color: #3D2B0E !important;
}

input[type="range"] { accent-color: #C9A84C !important; }
input[type="radio"] { accent-color: #C9A84C !important; }

footer { display: none !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #FEF9ED; }
::-webkit-scrollbar-thumb { background: #E0CFA0; border-radius: 3px; }
"""

# ── Shared constraint inputs builder ──────────────────────────
def constraint_row():
    with gr.Row():
        diet      = gr.Dropdown(DIET_OPTS,      label="Diet",      value="None")
        appliance = gr.Dropdown(APPLIANCE_OPTS, label="Appliance", value="None")
        time_lim  = gr.Dropdown(TIME_OPTS,      label="Time",      value="None")
        budget    = gr.Dropdown(BUDGET_OPTS,    label="Budget",    value="None")
    return diet, appliance, time_lim, budget

# ── UI ────────────────────────────────────────────────────────
with gr.Blocks(css=css, title="RecipeRAG") as demo:

    gr.HTML("""
    <div class="app-header">
        <h1>RecipeRAG</h1>
        <p>Context-aware recipe generation &nbsp;·&nbsp; RAG-powered &nbsp;·&nbsp; Indian kitchen friendly</p>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Text query ─────────────────────────────────
        with gr.Tab("What's in your kitchen"):
            with gr.Row():
                with gr.Column(scale=2):
                    t1_query = gr.Textbox(
                        label="What do you want to cook?",
                        placeholder="Something warm and filling with lentils...",
                        lines=2
                    )
                    t1_ingredients = gr.Textbox(
                        label="Ingredients you have  (comma separated)",
                        placeholder="tomato, onion, garlic, toor dal, cumin..."
                    )
                    t1_diet, t1_appliance, t1_time, t1_budget = constraint_row()
                    with gr.Row():
                        t1_servings = gr.Slider(1, 8, value=2, step=1, label="Servings")
                        t1_mode     = gr.Radio(MODE_OPTS, value="Advanced RAG (RRF)", label="Model mode")
                    t1_btn = gr.Button("Find my recipe →", variant="primary")

                with gr.Column(scale=3):
                    t1_recipe    = gr.Markdown(label="Recipe")
                    t1_retrieved = gr.Markdown(label="References")

            with gr.Row():
                t1_subs   = gr.Markdown(label="Substitutions")
                t1_reward = gr.Markdown(label="Quality score")
                t1_cost   = gr.Markdown(label="Cost estimate")

            t1_btn.click(
                fn=run_pipeline,
                inputs=[t1_query, t1_ingredients, t1_diet, t1_appliance, t1_time, t1_budget, t1_servings, t1_mode],
                outputs=[t1_recipe, t1_retrieved, t1_subs, t1_reward, t1_cost]
            )

        # ── Tab 2: Dish photo ─────────────────────────────────
        with gr.Tab("Got a photo?"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2_image = gr.Image(type="numpy", label="Upload a dish photo")
                    t2_detected = gr.Textbox(label="Detected dish", interactive=False)
                    t2_ingredients = gr.Textbox(label="Ingredients you have (optional)", placeholder="paneer, onion, tomato...")
                    t2_diet, t2_appliance, t2_time, t2_budget = constraint_row()
                    with gr.Row():
                        t2_servings = gr.Slider(1, 8, value=2, step=1, label="Servings")
                        t2_mode     = gr.Radio(MODE_OPTS, value="Advanced RAG (RRF)", label="Model mode")
                    t2_btn = gr.Button("Identify & generate →", variant="primary")

                with gr.Column(scale=2):
                    t2_recipe    = gr.Markdown(label="Recipe")
                    t2_retrieved = gr.Markdown(label="References")

            with gr.Row():
                t2_subs   = gr.Markdown(label="Substitutions")
                t2_reward = gr.Markdown(label="Quality score")
                t2_cost   = gr.Markdown(label="Cost estimate")

            t2_btn.click(
                fn=run_image_pipeline,
                inputs=[t2_image, t2_ingredients, t2_diet, t2_appliance, t2_time, t2_budget, t2_servings, t2_mode],
                outputs=[t2_detected, t2_recipe, t2_retrieved, t2_subs, t2_reward, t2_cost]
            )

        # ── Tab 3: Constraint builder ─────────────────────────
        with gr.Tab("Build from constraints"):
            with gr.Row():
                with gr.Column(scale=1):
                    t3_cuisine    = gr.Dropdown(CUISINE_OPTS, label="Cuisine", value="Any")
                    t3_diet, t3_appliance, t3_time, t3_budget = constraint_row()
                    t3_ingredients = gr.Textbox(label="Ingredients you have", placeholder="potato, onion, spices...")
                    with gr.Row():
                        t3_servings = gr.Slider(1, 8, value=2, step=1, label="Servings")
                        t3_mode     = gr.Radio(MODE_OPTS, value="Advanced RAG (RRF)", label="Model mode")
                    t3_btn = gr.Button("Find best recipe →", variant="primary")

                with gr.Column(scale=2):
                    t3_recipe    = gr.Markdown(label="Recipe")
                    t3_retrieved = gr.Markdown(label="References")

            with gr.Row():
                t3_subs   = gr.Markdown(label="Substitutions")
                t3_reward = gr.Markdown(label="Quality score")
                t3_cost   = gr.Markdown(label="Cost estimate")

            t3_btn.click(
                fn=run_constraint_pipeline,
                inputs=[t3_cuisine, t3_diet, t3_appliance, t3_time, t3_budget, t3_ingredients, t3_servings, t3_mode],
                outputs=[t3_recipe, t3_retrieved, t3_subs, t3_reward, t3_cost]
            )

# ── Launch ────────────────────────────────────────────────────
if __name__ == "__main__":
    is_kaggle = os.path.exists("/kaggle")
    demo.launch(share=is_kaggle, server_name="0.0.0.0", server_port=7860)
