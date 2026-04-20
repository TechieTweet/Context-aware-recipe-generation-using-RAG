# app/gradio_app.py
# Run: python app/gradio_app.py
# On Kaggle: set share=True at the bottom

import os
import sys
import gradio as gr

# ── Path setup — works whether run from repo root or app/ ──────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Person 1: Retrieval ────────────────────────────────────────
from retrieval.embedder import load_embedder
from retrieval.vector_store import load_vector_store
from retrieval.bm25_retriever import load_bm25_index
from retrieval.hybrid_retriever import hybrid_retrieve
import pandas as pd
import ast
import re
import pickle

# ── Person 2: LLM generation ───────────────────────────────────
from generation.llm import generate_recipe, build_prompt

# ── Person 3: Substitution + Reward + Cost ────────────────────
# Load using importlib (Person 3 uses relative imports internally)
import importlib.util

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_p3_pipeline = _load_module(
    "person3_pipeline",
    os.path.join(ROOT, "person3", "pipeline.py")
)
person3_pipeline = _p3_pipeline.person3_pipeline


# ══════════════════════════════════════════════════════════════
# LOAD ALL COMPONENTS AT STARTUP
# ══════════════════════════════════════════════════════════════

print("=" * 55)
print("Loading RecipeRAG components...")
print("=" * 55)

# Paths — adjust BASE_DIR if running locally vs Kaggle
BASE_DIR = os.environ.get("RECIPE_BASE_DIR", os.path.join(ROOT, "data"))

print("Loading embedder...")
embedder = load_embedder()

print("Loading ChromaDB...")
collection = load_vector_store()

print("Loading BM25 index...")
bm25 = load_bm25_index()

print("Loading DataFrame...")
df = pd.read_parquet(os.path.join(BASE_DIR, "recipes_clean.parquet"))

# Re-parse ingredients and directions if stored as strings
def safe_parse(val):
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except Exception:
        return []

df["ingredients"] = df["ingredients"].apply(safe_parse)
df["directions"]  = df["directions"].apply(safe_parse)

def build_doc(row):
    return f"{row['title']}. Ingredients: {', '.join(row['ingredients'])}"

def build_full_text(row):
    ing  = "\n".join(f"- {i}" for i in row["ingredients"])
    dirs = "\n".join(f"{idx+1}. {s}" for idx, s in enumerate(row["directions"]))
    return f"# {row['title']}\n\n## Ingredients\n{ing}\n\n## Instructions\n{dirs}"

if "document" not in df.columns:
    df["document"]  = df.apply(build_doc, axis=1)
if "full_text" not in df.columns:
    df["full_text"] = df.apply(build_full_text, axis=1)

print("All components loaded. Starting Gradio app...\n")


# ══════════════════════════════════════════════════════════════
# CORE PIPELINE FUNCTION
# Called by all three Gradio tabs
# ══════════════════════════════════════════════════════════════

def run_pipeline(
    query: str,
    ingredients_text: str = "",
    diet: str = "None",
    appliance: str = "None",
    time_limit: str = "None",
    budget: str = "None",
    servings: int = 2
):
    """
    Full pipeline: retrieve → generate → evaluate + substitute + cost.
    Returns formatted strings for each Gradio output component.
    """
    if not query.strip():
        return "Please enter a query.", "", "", "", ""

    # ── Build constraints dict ─────────────────────────────────
    available_ingredients = [
        i.strip() for i in ingredients_text.split(",") if i.strip()
    ] if ingredients_text.strip() else []

    constraints = {}
    if available_ingredients:
        constraints["ingredients"] = available_ingredients
    if diet and diet != "None":
        constraints["diet"] = diet
    if appliance and appliance != "None":
        constraints["appliance"] = appliance
    if time_limit and time_limit != "None":
        constraints["time"] = time_limit
    if budget and budget != "None":
        constraints["budget"] = budget

    # ── Step 1: Hybrid retrieval (Person 1) ───────────────────
    try:
        retrieved = hybrid_retrieve(
            query=query,
            df=df,
            bm25=bm25,
            collection=collection,
            embedder=embedder,
            top_k=5
        )
    except Exception as e:
        return f"Retrieval error: {str(e)}", "", "", "", ""

    retrieved_titles = "\n".join(
        f"{i+1}. {r['title']}" for i, r in enumerate(retrieved)
    )

    # ── Step 2: LLM generation (Person 2) ─────────────────────
    try:
        generated_recipe = generate_recipe(
            query=query,
            retrieved_recipes=retrieved,
            constraints=constraints
        )
    except Exception as e:
        generated_recipe = f"Generation error: {str(e)}"

    # ── Step 3: Parse generated recipe for Person 3 ───────────
    # Extract ingredients and steps from generated text
    recipe_lines = generated_recipe.split("\n")

    gen_ingredients = []
    gen_steps = []
    in_ingredients = False
    in_instructions = False

    for line in recipe_lines:
        line = line.strip()
        if "ingredients:" in line.lower():
            in_ingredients = True
            in_instructions = False
            continue
        if "instructions:" in line.lower() or "directions:" in line.lower():
            in_instructions = True
            in_ingredients = False
            continue
        if "estimated time:" in line.lower() or "serves:" in line.lower():
            in_instructions = False
            in_ingredients = False
            continue
        if in_ingredients and line.startswith("-"):
            gen_ingredients.append(line.lstrip("- ").strip())
        if in_instructions and re.match(r"^\d+\.", line):
            gen_steps.append(re.sub(r"^\d+\.\s*", "", line).strip())

    # Fallback if parsing fails
    if not gen_ingredients:
        gen_ingredients = available_ingredients or ["unknown"]
    if not gen_steps:
        gen_steps = [generated_recipe[:200]]

    # Estimate time from generated text
    est_time = None
    time_match = re.search(r"estimated time[:\s]+(\d+)", generated_recipe.lower())
    if time_match:
        est_time = int(time_match.group(1))

    max_time = None
    if time_limit and time_limit != "None":
        t_match = re.search(r"(\d+)", time_limit)
        if t_match:
            max_time = int(t_match.group(1))

    # ── Step 4: Substitution + Reward + Cost (Person 3) ───────
    try:
        p3_results = person3_pipeline(
            recipe_steps=gen_steps,
            recipe_ingredients=gen_ingredients,
            available_ingredients=available_ingredients if available_ingredients else gen_ingredients,
            dietary_restrictions=[diet.lower()] if diet and diet != "None" else None,
            available_appliances=[appliance.lower()] if appliance and appliance != "None" else None,
            max_time_minutes=max_time,
            estimated_time_minutes=est_time,
            servings=servings,
            top_k_substitutes=3
        )

        # Format substitutions
        subs_text = ""
        if p3_results.get("missing_ingredients"):
            subs_text += f"**Missing ingredients:** {', '.join(p3_results['missing_ingredients'])}\n\n"
            subs_text += "**Suggested substitutes:**\n"
            for ing, subs in p3_results["substitutions"].items():
                subs_text += f"\n🔄 **{ing}** →\n"
                for s in subs:
                    tag = "🇮🇳" if s["is_indian"] else "🌍"
                    subs_text += f"  {tag} {s['ingredient']} (score: {s['final_score']})\n"
        else:
            subs_text = "✅ All ingredients available — no substitutions needed!"

        # Format reward score
        r = p3_results["reward"]
        reward_text = (
            f"**{p3_results['verdict']}**\n\n"
            f"| Metric | Score |\n"
            f"|--------|-------|\n"
            f"| Coherence (30%) | {r['coherence_score']} |\n"
            f"| Constraint Satisfaction (40%) | {r['constraint_satisfaction_score']} |\n"
            f"| Ingredient Feasibility (30%) | {r['ingredient_feasibility_score']} |\n"
            f"| **Final Reward** | **{r['final_reward']}** |"
        )

        # Format cost
        c = p3_results["cost"]
        cost_text = (
            f"**{c['budget_category']}**\n\n"
            f"Total: {c['total_cost']}  |  Per serving: {c['cost_per_serving']}\n\n"
            f"**Breakdown:**\n"
        )
        for item in c["breakdown"]:
            cost_text += f"- {item['ingredient']}: {item['cost']} ({item['unit_price']})\n"
        if c.get("ingredients_not_found"):
            cost_text += f"\n*Not in price DB: {', '.join(c['ingredients_not_found'])}*"

    except Exception as e:
        subs_text  = f"Substitution error: {str(e)}"
        reward_text = "Reward scoring unavailable."
        cost_text  = "Cost estimation unavailable."

    return (
        generated_recipe,
        f"**Retrieved recipes:**\n{retrieved_titles}",
        subs_text,
        reward_text,
        cost_text
    )


# ══════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════

DIET_OPTIONS     = ["None", "Vegetarian", "Vegan", "Jain", "Gluten Free", "Diabetic"]
APPLIANCE_OPTIONS = ["None", "Stovetop", "Pressure Cooker", "Microwave", "Oven", "Air Fryer"]
TIME_OPTIONS     = ["None", "15 minutes", "30 minutes", "45 minutes", "60 minutes", "90 minutes"]
BUDGET_OPTIONS   = ["None", "Under ₹50", "Under ₹150", "Under ₹300", "No limit"]

with gr.Blocks(title="RecipeRAG", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🍳 RecipeRAG — Context-Aware Recipe Generation
    *Retrieval-Augmented Generation with ingredient substitution and cost estimation*
    """)

    with gr.Tabs():

        # ── Tab 1: Free-form text query ───────────────────────
        with gr.Tab("📝 Text Query"):
            gr.Markdown("Type what you have or what you're craving.")

            with gr.Row():
                with gr.Column(scale=2):
                    t1_query = gr.Textbox(
                        label="Query",
                        placeholder="e.g. I have tomatoes, onion, garlic and rice. What can I make?",
                        lines=2
                    )
                    t1_ingredients = gr.Textbox(
                        label="Available Ingredients (comma separated, optional)",
                        placeholder="tomato, onion, garlic, rice, salt"
                    )
                    with gr.Row():
                        t1_diet      = gr.Dropdown(DIET_OPTIONS, label="Dietary restriction", value="None")
                        t1_appliance = gr.Dropdown(APPLIANCE_OPTIONS, label="Appliance", value="None")
                        t1_time      = gr.Dropdown(TIME_OPTIONS, label="Time limit", value="None")
                        t1_budget    = gr.Dropdown(BUDGET_OPTIONS, label="Budget", value="None")
                    t1_servings = gr.Slider(1, 8, value=2, step=1, label="Servings")
                    t1_btn = gr.Button("Generate Recipe 🍽️", variant="primary")

            with gr.Row():
                with gr.Column():
                    t1_recipe   = gr.Markdown(label="Generated Recipe")
                with gr.Column():
                    t1_retrieved = gr.Markdown(label="Retrieved References")

            with gr.Row():
                with gr.Column():
                    t1_subs   = gr.Markdown(label="Substitutions")
                with gr.Column():
                    t1_reward = gr.Markdown(label="Quality Score")
                with gr.Column():
                    t1_cost   = gr.Markdown(label="Cost Estimate")

            t1_btn.click(
                fn=run_pipeline,
                inputs=[t1_query, t1_ingredients, t1_diet, t1_appliance, t1_time, t1_budget, t1_servings],
                outputs=[t1_recipe, t1_retrieved, t1_subs, t1_reward, t1_cost]
            )

        # ── Tab 2: Image input (CLIP) ─────────────────────────
        with gr.Tab("📷 Dish Photo"):
            gr.Markdown("Upload a photo of a dish — we'll identify it and find matching recipes.")

            with gr.Row():
                with gr.Column(scale=1):
                    t2_image = gr.Image(type="pil", label="Upload dish photo")
                    t2_ingredients = gr.Textbox(
                        label="Available Ingredients (optional)",
                        placeholder="tomato, onion, garlic"
                    )
                    with gr.Row():
                        t2_diet      = gr.Dropdown(DIET_OPTIONS, label="Dietary restriction", value="None")
                        t2_appliance = gr.Dropdown(APPLIANCE_OPTIONS, label="Appliance", value="None")
                        t2_time      = gr.Dropdown(TIME_OPTIONS, label="Time limit", value="None")
                        t2_budget    = gr.Dropdown(BUDGET_OPTIONS, label="Budget", value="None")
                    t2_servings  = gr.Slider(1, 8, value=2, step=1, label="Servings")
                    t2_btn = gr.Button("Identify & Generate 🔍", variant="primary")

            t2_detected = gr.Textbox(label="Detected dish", interactive=False)

            with gr.Row():
                with gr.Column():
                    t2_recipe    = gr.Markdown(label="Generated Recipe")
                with gr.Column():
                    t2_retrieved = gr.Markdown(label="Retrieved References")

            with gr.Row():
                with gr.Column():
                    t2_subs   = gr.Markdown(label="Substitutions")
                with gr.Column():
                    t2_reward = gr.Markdown(label="Quality Score")
                with gr.Column():
                    t2_cost   = gr.Markdown(label="Cost Estimate")

            def run_image_pipeline(image, ingredients_text, diet, appliance, time_limit, budget, servings):
                if image is None:
                    return "No image uploaded.", "", "", "", "", ""
                try:
                    from transformers import CLIPProcessor, CLIPModel
                    import torch

                    DISH_LABELS = [
                        "dal tadka", "paneer butter masala", "biryani", "pasta",
                        "pizza", "fried rice", "chicken curry", "idli sambar",
                        "dosa", "burger", "salad", "soup", "noodles", "sandwich",
                        "chocolate cake", "rajma chawal", "aloo paratha", "pulao"
                    ]

                    clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

                    inputs = clip_processor(
                        text=DISH_LABELS,
                        images=image,
                        return_tensors="pt",
                        padding=True
                    )
                    with torch.no_grad():
                        outputs    = clip_model(**inputs)
                        probs      = outputs.logits_per_image.softmax(dim=1)
                        best_idx   = probs.argmax().item()
                        detected   = DISH_LABELS[best_idx]
                        confidence = round(probs[0][best_idx].item() * 100, 1)

                    query = f"{detected} recipe"
                    detected_str = f"{detected} ({confidence}% confidence)"

                except Exception as e:
                    detected_str = f"Could not identify dish: {str(e)}"
                    query = "recipe"

                recipe, retrieved, subs, reward, cost = run_pipeline(
                    query, ingredients_text, diet, appliance, time_limit, budget, servings
                )
                return detected_str, recipe, retrieved, subs, reward, cost

            t2_btn.click(
                fn=run_image_pipeline,
                inputs=[t2_image, t2_ingredients, t2_diet, t2_appliance, t2_time, t2_budget, t2_servings],
                outputs=[t2_detected, t2_recipe, t2_retrieved, t2_subs, t2_reward, t2_cost]
            )

        # ── Tab 3: Constraint dropdowns ───────────────────────
        with gr.Tab("🎛️ Constraint Builder"):
            gr.Markdown("Select your constraints and let the system suggest a recipe.")

            with gr.Row():
                with gr.Column():
                    t3_cuisine    = gr.Dropdown(
                        ["Any", "Indian", "Italian", "Chinese", "Mexican", "Continental", "South Indian", "Mughlai"],
                        label="Cuisine", value="Any"
                    )
                    t3_diet       = gr.Dropdown(DIET_OPTIONS, label="Dietary restriction", value="None")
                    t3_appliance  = gr.Dropdown(APPLIANCE_OPTIONS, label="Appliance", value="None")
                    t3_time       = gr.Dropdown(TIME_OPTIONS, label="Time limit", value="None")
                    t3_budget     = gr.Dropdown(BUDGET_OPTIONS, label="Budget", value="None")
                    t3_ingredients = gr.Textbox(
                        label="Available Ingredients",
                        placeholder="tomato, paneer, onion, garam masala"
                    )
                    t3_servings   = gr.Slider(1, 8, value=2, step=1, label="Servings")
                    t3_btn        = gr.Button("Find Best Recipe 🎯", variant="primary")

            with gr.Row():
                with gr.Column():
                    t3_recipe    = gr.Markdown(label="Generated Recipe")
                with gr.Column():
                    t3_retrieved = gr.Markdown(label="Retrieved References")

            with gr.Row():
                with gr.Column():
                    t3_subs   = gr.Markdown(label="Substitutions")
                with gr.Column():
                    t3_reward = gr.Markdown(label="Quality Score")
                with gr.Column():
                    t3_cost   = gr.Markdown(label="Cost Estimate")

            def run_constraint_pipeline(cuisine, diet, appliance, time_limit, budget, ingredients_text, servings):
                cuisine_str = f"{cuisine} cuisine" if cuisine != "Any" else ""
                diet_str    = f"that is {diet.lower()}" if diet != "None" else ""
                time_str    = f"ready in {time_limit}" if time_limit != "None" else ""
                parts       = [p for p in [cuisine_str, diet_str, time_str] if p]
                query       = "Give me a recipe " + " and ".join(parts) if parts else "Give me a good recipe"
                if ingredients_text.strip():
                    query += f" using: {ingredients_text}"

                return run_pipeline(query, ingredients_text, diet, appliance, time_limit, budget, servings)

            t3_btn.click(
                fn=run_constraint_pipeline,
                inputs=[t3_cuisine, t3_diet, t3_appliance, t3_time, t3_budget, t3_ingredients, t3_servings],
                outputs=[t3_recipe, t3_retrieved, t3_subs, t3_reward, t3_cost]
            )


# ══════════════════════════════════════════════════════════════
# LAUNCH
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # share=True gives a public link — required for Kaggle
    # share=False for local testing
    is_kaggle = os.path.exists("/kaggle")
    demo.launch(share=is_kaggle, server_name="0.0.0.0", server_port=7860)
