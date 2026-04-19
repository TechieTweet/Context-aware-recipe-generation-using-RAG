# app/gradio_app.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from PIL import Image

from config import DATAFRAME_PATH
from retrieval.embedder import load_embedder
from retrieval.vector_store import load_vector_store
from retrieval.bm25_retriever import load_bm25_index
from retrieval.hybrid_retriever import hybrid_retrieve
from generation.llm import generate_recipe
from app.clip_classifier import load_clip_model, classify_dish

import pandas as pd

# ── Load all models and indexes at startup ──────────────────────
print("Loading embedder...")
embedder = load_embedder()

print("Loading ChromaDB...")
collection = load_vector_store()

print("Loading BM25 index...")
bm25 = load_bm25_index()

print("Loading recipes dataframe...")
df = pd.read_parquet(DATAFRAME_PATH)

print("Loading CLIP model...")
clip_model, clip_processor, clip_device = load_clip_model()

print("All models loaded. Starting Gradio app...")


# ── Core pipeline function ──────────────────────────────────────
def run_pipeline(
    text_query,
    dish_image,
    available_ingredients,
    diet,
    appliance,
    time_limit,
    budget
):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return "❌ HF_TOKEN not set. Add it to your .env file.", ""

    # ── Step 1: Determine query ──
    if dish_image is not None:
        pil_image = Image.fromarray(dish_image).convert("RGB")
        dish_name = classify_dish(pil_image, clip_model, clip_processor, clip_device)
        query = dish_name
        query_display = f"📷 Detected dish: **{dish_name}**"
        if text_query.strip():
            query = f"{dish_name} {text_query.strip()}"
            query_display += f" + your notes: *{text_query.strip()}*"
    elif text_query.strip():
        query = text_query.strip()
        query_display = f"🔍 Query: **{query}**"
    else:
        return "⚠️ Please enter a query or upload a dish photo.", ""

    # ── Step 2: Build constraints dict ──
    constraints = {}

    if available_ingredients.strip():
        constraints["ingredients"] = [
            i.strip() for i in available_ingredients.split(",") if i.strip()
        ]

    if diet and diet != "None":
        constraints["diet"] = diet

    if appliance and appliance != "None":
        constraints["appliance"] = appliance

    if time_limit.strip():
        constraints["time"] = time_limit.strip()

    if budget.strip():
        constraints["budget"] = budget.strip()

    # ── Step 3: Retrieve ──
    try:
        retrieved = hybrid_retrieve(
            query=query,
            df=df,
            bm25=bm25,
            collection=collection,
            embedder=embedder
        )
    except Exception as e:
        return query_display, f"❌ Retrieval error: {str(e)}"

    if not retrieved:
        return query_display, "❌ No relevant recipes found. Try a different query."

    retrieved_titles = "\n".join([f"- {r['title']}" for r in retrieved])
    retrieval_info = f"{query_display}\n\n📚 Retrieved references:\n{retrieved_titles}"

    # ── Step 4: Generate ──
    try:
        recipe_output = generate_recipe(
            query=query,
            retrieved_recipes=retrieved,
            hf_token=hf_token,
            constraints=constraints
        )
    except Exception as e:
        return retrieval_info, f"❌ Generation error: {str(e)}"

    return retrieval_info, recipe_output


# ── Gradio UI ───────────────────────────────────────────────────
with gr.Blocks(title="RecipeRAG") as demo:

    gr.Markdown("""
    # 🍳 RecipeRAG — Context-Aware Recipe Generation
    Generate personalised recipes using Retrieval-Augmented Generation.
    Provide a text query **or** upload a dish photo. Add constraints to personalise further.
    """)

    with gr.Row():

        # ── Left column: Inputs ──
        with gr.Column(scale=1):

            gr.Markdown("### 🍽️ What do you want to cook?")
            text_query = gr.Textbox(
                label="Text Query",
                placeholder="e.g. quick vegetarian dinner with paneer and spinach",
                lines=2
            )
            dish_image = gr.Image(
                label="Or upload a dish photo (optional)",
                type="numpy"
            )

            gr.Markdown("### ⚙️ Constraints (all optional)")
            available_ingredients = gr.Textbox(
                label="Available Ingredients (comma-separated)",
                placeholder="e.g. paneer, spinach, onion, tomato, garam masala"
            )
            diet = gr.Dropdown(
                label="Dietary Restriction",
                choices=["None", "Vegetarian", "Vegan", "Gluten-free", "Jain", "Halal"],
                value="None"
            )
            appliance = gr.Dropdown(
                label="Appliance",
                choices=["None", "Stovetop only", "Microwave only", "Oven", "Air fryer", "Pressure cooker"],
                value="None"
            )
            time_limit = gr.Textbox(
                label="Time Limit",
                placeholder="e.g. 30 minutes"
            )
            budget = gr.Textbox(
                label="Budget",
                placeholder="e.g. under ₹200"
            )

            submit_btn = gr.Button("🚀 Generate Recipe", variant="primary")

        # ── Right column: Outputs ──
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Retrieval Info")
            retrieval_output = gr.Markdown()

            gr.Markdown("### 🍴 Generated Recipe")
            recipe_output = gr.Textbox(
                label="Recipe",
                lines=20
            )

    submit_btn.click(
        fn=run_pipeline,
        inputs=[
            text_query,
            dish_image,
            available_ingredients,
            diet,
            appliance,
            time_limit,
            budget
        ],
        outputs=[retrieval_output, recipe_output]
    )

    gr.Markdown("""
    ---
    **Team:** Person 1 — RAG pipeline | Person 2 — LLM + CLIP + UI | Person 3 — Substitution + Cost
    """)


if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft())