# app/gradio_app.py — Integrated with Person 3 + Themed UI
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from PIL import Image
import re

from config import DATAFRAME_PATH
from retrieval.embedder import load_embedder
from retrieval.vector_store import load_vector_store
from retrieval.bm25_retriever import load_bm25_index
from retrieval.hybrid_retriever import hybrid_retrieve
from generation.llm import generate_recipe
from app.clip_classifier import load_clip_model, classify_dish
from substitution.substitutor import get_substitutes, estimate_cost_inr, evaluate_recipe

import pandas as pd

# ── Load all models at startup ─────────────────────────────────
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


# ── CSS matching the slide design ─────────────────────────────
CUSTOM_CSS = """
/* ── Global background — warm cream like the slides ── */
body, .gradio-container {
    background-color: #FFF8EE !important;
    font-family: 'Georgia', serif !important;
}

/* ── Tab bar ── */
.tab-nav button {
    background-color: #FFF0D6 !important;
    color: #7B3F00 !important;
    font-weight: bold !important;
    border-radius: 12px 12px 0 0 !important;
    font-size: 15px !important;
    border: 2px solid #E8C99A !important;
}
.tab-nav button.selected {
    background-color: #C8601A !important;
    color: white !important;
    border-color: #C8601A !important;
}

/* ── Cards / panels ── */
.gr-box, .gr-form, .gr-panel, .block {
    background-color: #FFF5E6 !important;
    border: 1.5px solid #E8C99A !important;
    border-radius: 16px !important;
}

/* ── Labels ── */
label span, .gr-form label {
    color: #7B3F00 !important;
    font-weight: bold !important;
    font-size: 14px !important;
}

/* ── Primary button — dark orange like slide accents ── */
.gr-button-primary, button.primary {
    background: linear-gradient(135deg, #C8601A, #A0420D) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    font-size: 16px !important;
    font-weight: bold !important;
    padding: 12px 30px !important;
    box-shadow: 0 4px 12px rgba(200, 96, 26, 0.4) !important;
}
.gr-button-primary:hover {
    background: linear-gradient(135deg, #A0420D, #7B3F00) !important;
    transform: translateY(-1px) !important;
}

/* ── Secondary button ── */
.gr-button-secondary {
    background-color: #FFF0D6 !important;
    color: #7B3F00 !important;
    border: 2px solid #C8601A !important;
    border-radius: 25px !important;
    font-weight: bold !important;
}

/* ── Textbox inputs ── */
textarea, input[type=text], input[type=number] {
    background-color: #FFFDF7 !important;
    border: 1.5px solid #D4A96A !important;
    border-radius: 10px !important;
    color: #3D1C00 !important;
    font-size: 14px !important;
}

/* ── Dropdowns ── */
select, .gr-dropdown {
    background-color: #FFFDF7 !important;
    border: 1.5px solid #D4A96A !important;
    border-radius: 10px !important;
    color: #3D1C00 !important;
}

/* ── Markdown headings ── */
.gr-markdown h1 {
    color: #7B3F00 !important;
    font-size: 2.2em !important;
    font-weight: 900 !important;
    text-align: center !important;
    text-shadow: 1px 1px 0px #E8C99A !important;
}
.gr-markdown h2 {
    color: #7B3F00 !important;
    font-size: 1.5em !important;
    font-weight: bold !important;
    border-bottom: 2px solid #E8C99A !important;
    padding-bottom: 4px !important;
}
.gr-markdown h3 {
    color: #C8601A !important;
    font-size: 1.2em !important;
    font-weight: bold !important;
}

/* ── Slider ── */
input[type=range] {
    accent-color: #C8601A !important;
}

/* ── Checkbox group ── */
input[type=checkbox] {
    accent-color: #C8601A !important;
}

/* ── Image upload area ── */
.gr-image {
    border: 2px dashed #D4A96A !important;
    border-radius: 16px !important;
    background-color: #FFF5E6 !important;
}

/* ── Footer ── */
.footer-text {
    text-align: center !important;
    color: #A06030 !important;
    font-size: 13px !important;
    padding: 10px !important;
}
"""

# ── Helper: Parse generated recipe text ───────────────────────
def parse_recipe_output(recipe_text: str):
    ingredients = []
    steps = []

    ing_match = re.search(
        r"Ingredients:\s*(.*?)(?=Instructions:|$)",
        recipe_text, re.DOTALL | re.IGNORECASE
    )
    if ing_match:
        for line in ing_match.group(1).strip().split("\n"):
            line = line.strip().lstrip("-•*").strip()
            if line:
                ingredients.append(line)

    inst_match = re.search(
        r"Instructions:\s*(.*?)(?=Estimated Time:|Serves:|$)",
        recipe_text, re.DOTALL | re.IGNORECASE
    )
    if inst_match:
        for line in inst_match.group(1).strip().split("\n"):
            line = re.sub(r"^\d+\.\s*", "", line.strip()).strip()
            if line:
                steps.append(line)

    time_match = re.search(r"Estimated Time:\s*(.+)", recipe_text, re.IGNORECASE)
    estimated_time = time_match.group(1).strip() if time_match else ""

    return ingredients, steps, estimated_time


# ── Core pipeline ──────────────────────────────────────────────
def run_pipeline(
    text_query, dish_image, available_ingredients,
    diet, appliance, time_limit, budget
):
    hf_token = os.environ.get("HF_TOKEN", "")

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
        return "⚠️ Please enter a query or upload a dish photo.", "", "", "", ""

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

    try:
        retrieved = hybrid_retrieve(
            query=query, df=df, bm25=bm25,
            collection=collection, embedder=embedder
        )
    except Exception as e:
        return query_display, f"❌ Retrieval error: {str(e)}", "", "", ""

    if not retrieved:
        return query_display, "❌ No relevant recipes found.", "", "", ""

    retrieved_titles = "\n".join([f"🍽️ {r['title']}" for r in retrieved])
    retrieval_info = f"{query_display}\n\n📚 **Retrieved References:**\n{retrieved_titles}"

    try:
        recipe_output = generate_recipe(
            query=query, retrieved_recipes=retrieved,
            hf_token=hf_token, constraints=constraints
        )
    except Exception as e:
        return retrieval_info, f"❌ Generation error: {str(e)}", "", "", ""

    # Person 3 — Cost + Evaluation
    try:
        avail_list = [i.strip() for i in available_ingredients.split(",") if i.strip()]
        diet_list = [diet.lower()] if diet and diet != "None" else []
        appliance_list = [appliance.lower()] if appliance and appliance != "None" else []

        parsed_ingredients, parsed_steps, _ = parse_recipe_output(recipe_output)

        cost_result = estimate_cost_inr(parsed_ingredients, servings=2)
        cost_display = (
            f"## {cost_result.get('budget_category', '💰 Cost Estimate')}\n\n"
            f"**Total Cost:** {cost_result.get('total_cost', '₹?')}  "
            f"**Per Serving:** {cost_result.get('cost_per_serving', '₹?')}\n\n"
            f"**Breakdown:**\n"
        )
        for item in cost_result.get("breakdown", []):
            cost_display += f"- {item['ingredient']}: {item['cost']}\n"

        eval_result = evaluate_recipe(
            recipe_steps=parsed_steps,
            recipe_ingredients=parsed_ingredients,
            available_ingredients=avail_list if avail_list else parsed_ingredients,
            dietary_restrictions=diet_list if diet_list else None,
            available_appliances=appliance_list if appliance_list else None,
            estimated_time_minutes=30
        )

        score = eval_result.get("final_reward", 0)
        if score >= 0.75:
            verdict = f"✅ Excellent Recipe!"
        elif score >= 0.55:
            verdict = f"⚠️ Good Recipe"
        elif score >= 0.35:
            verdict = f"🟠 Fair Recipe"
        else:
            verdict = f"❌ Needs Improvement"

        eval_display = (
            f"## {verdict}  Score: {score}\n\n"
            f"| Metric | Score |\n"
            f"|--------|-------|\n"
            f"| 🧠 Coherence | {eval_result.get('coherence_score', 'N/A')} |\n"
            f"| ✅ Constraints | {eval_result.get('constraint_satisfaction_score', 'N/A')} |\n"
            f"| 🥘 Feasibility | {eval_result.get('ingredient_feasibility_score', 'N/A')} |\n"
            f"| 🏆 Final Reward | **{score}** |\n"
        )

    except Exception as e:
        cost_display = f"Cost estimation unavailable: {str(e)}"
        eval_display = f"Evaluation unavailable: {str(e)}"
        parsed_ingredients = []

    return (retrieval_info, recipe_output, cost_display,
            eval_display, ", ".join(parsed_ingredients))


# ── Substitution ───────────────────────────────────────────────
def run_substitution(ingredient_query, top_k):
    if not ingredient_query.strip():
        return "⚠️ Please enter an ingredient or query."

    query = ingredient_query.lower().strip()
    patterns = [
        r"(?:vegan|vegetarian|jain|gluten.free|dairy.free)\s+alternative\s+to\s+(.+)",
        r"substitute\s+for\s+(.+)",
        r"replace\s+(.+)",
        r"instead\s+of\s+(.+)",
        r"alternative\s+to\s+(.+)",
        r"(.+)\s+substitute",
    ]
    ingredient = query
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            ingredient = match.group(1).strip()
            break

    subs = get_substitutes(ingredient, top_k=int(top_k))
    if not subs:
        return f"No substitutes found for **{ingredient}**"

    output = f"### 🔄 Top substitutes for **{ingredient}**:\n\n"
    for i, s in enumerate(subs, 1):
        tag = "🇮🇳 Indian" if s.get("is_indian") else "🌍 Global"
        src = "📖 Curated" if s.get("source") == "curated_kb" else "🤖 AI"
        output += (
            f"**{i}. {s['ingredient'].title()}** — {tag} · {src}\n"
            f"> Flavor match: `{s.get('flavor_similarity', 'N/A')}` | "
            f"Semantic: `{s.get('embedding_similarity', 'N/A')}` | "
            f"Score: `{s.get('final_score', 'N/A')}`\n\n"
        )
    return output


# ── Cost standalone ────────────────────────────────────────────
def run_cost_estimator(ingredients_text, servings):
    ings = [i.strip() for i in ingredients_text.strip().split("\n") if i.strip()]
    if not ings:
        return "⚠️ Please enter at least one ingredient."
    result = estimate_cost_inr(ings, servings=int(servings))
    output = (
        f"## {result.get('budget_category', '')}\n\n"
        f"**Total Cost:** {result.get('total_cost', '₹?')} &nbsp;&nbsp; "
        f"**Per Serving:** {result.get('cost_per_serving', '₹?')}\n\n"
        f"---\n\n**Ingredient Breakdown:**\n\n"
    )
    for item in result.get("breakdown", []):
        output += f"- **{item['ingredient']}**: {item['cost']} *(unit: {item['unit_price']})*\n"
    if result.get("ingredients_not_found"):
        output += f"\n⚠️ Prices not found for: `{', '.join(result['ingredients_not_found'])}`"
    return output


# ── Evaluator standalone ───────────────────────────────────────
def run_evaluator(steps_text, ing_text, avail_text,
                  diet_list, appliance_list, max_time, est_time):
    steps = [s.strip() for s in steps_text.strip().split("\n") if s.strip()]
    ings = [i.strip() for i in ing_text.strip().split("\n") if i.strip()]
    avail = [i.strip() for i in avail_text.strip().split("\n") if i.strip()]

    if not steps or not ings or not avail:
        return "⚠️ Please fill in steps, ingredients, and available ingredients."

    result = evaluate_recipe(
        recipe_steps=steps,
        recipe_ingredients=ings,
        available_ingredients=avail,
        dietary_restrictions=diet_list if diet_list else None,
        available_appliances=appliance_list if appliance_list else None,
        max_time_minutes=int(max_time) if max_time > 0 else None,
        estimated_time_minutes=int(est_time) if est_time > 0 else None
    )

    score = result.get("final_reward", 0)
    if score >= 0.75:
        verdict = "✅ Excellent Recipe!"
    elif score >= 0.55:
        verdict = "⚠️ Good Recipe"
    elif score >= 0.35:
        verdict = "🟠 Fair Recipe"
    else:
        verdict = "❌ Needs Improvement"

    return (
        f"## {verdict}\n\n"
        f"**Final Reward Score: {score}**\n\n"
        f"---\n\n"
        f"| Metric | Score |\n"
        f"|--------|-------|\n"
        f"| 🧠 Coherence | {result.get('coherence_score', 'N/A')} |\n"
        f"| ✅ Constraint Satisfaction | {result.get('constraint_satisfaction_score', 'N/A')} |\n"
        f"| 🥘 Ingredient Feasibility | {result.get('ingredient_feasibility_score', 'N/A')} |\n"
        f"| 🏆 **Final Reward** | **{score}** |\n"
    )


# ── Gradio UI ──────────────────────────────────────────────────
with gr.Blocks(
    title="RecipeRAG",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("""
    # 🍳 RecipeRAG — Context-Aware Recipe Generation
    Generate personalised recipes using Retrieval-Augmented Generation.
    Provide a text query **or** upload a dish photo. Add constraints to personalise further.
    """)

    with gr.Tabs():

        # ════════════════════════════════════════════════════
        # TAB 1 — Generate Recipe
        # ════════════════════════════════════════════════════
        with gr.Tab("🍴 Generate Recipe"):
            gr.Markdown("### 🍽️ What do you want to cook?")

            with gr.Row():
                with gr.Column(scale=1):

                    text_query = gr.Textbox(
                        label="Your Query",
                        placeholder="e.g. quick vegetarian dal with spinach under 30 minutes",
                        lines=2
                    )
                    dish_image = gr.Image(
                        label="📷 Upload a Dish Photo (optional — we'll detect it!)",
                        type="numpy"
                    )

                    gr.Markdown("### ⚙️ Your Constraints")
                    available_ingredients = gr.Textbox(
                        label="🧺 Available Ingredients (comma-separated)",
                        placeholder="paneer, spinach, onion, tomato, garam masala, ghee"
                    )

                    with gr.Row():
                        diet = gr.Dropdown(
                            label="🥗 Dietary Restriction",
                            choices=["None", "Vegetarian", "Vegan",
                                     "Gluten-free", "Jain", "Halal"],
                            value="None"
                        )
                        appliance = gr.Dropdown(
                            label="🍳 Appliance",
                            choices=["None", "Stovetop only", "Microwave only",
                                     "Oven", "Air fryer", "Pressure cooker"],
                            value="None"
                        )

                    with gr.Row():
                        time_limit = gr.Textbox(
                            label="⏱️ Time Limit",
                            placeholder="e.g. 30 minutes"
                        )
                        budget = gr.Textbox(
                            label="💰 Budget",
                            placeholder="e.g. under ₹200"
                        )

                    submit_btn = gr.Button(
                        "🚀 Generate My Recipe!",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 📚 Retrieved References")
                    retrieval_output = gr.Markdown(
                        value="*Your retrieval info will appear here...*"
                    )

                    gr.Markdown("### 🍴 Your Personalised Recipe")
                    recipe_output = gr.Textbox(
                        label="Generated Recipe",
                        lines=18,
                        placeholder="Your recipe will appear here after generation..."
                    )

                    parsed_ingredients_state = gr.Textbox(
                        label="📝 Parsed Ingredients",
                        interactive=False,
                        visible=True
                    )

            gr.Markdown("### 💸 Cost Estimate & 📊 Recipe Evaluation")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 💸 Cost Breakdown")
                    cost_output = gr.Markdown(
                        value="*Cost estimate will appear after recipe generation...*"
                    )
                with gr.Column():
                    gr.Markdown("### 📊 Recipe Quality Score")
                    eval_output = gr.Markdown(
                        value="*Quality evaluation will appear after recipe generation...*"
                    )

            submit_btn.click(
                fn=run_pipeline,
                inputs=[text_query, dish_image, available_ingredients,
                        diet, appliance, time_limit, budget],
                outputs=[retrieval_output, recipe_output,
                         cost_output, eval_output, parsed_ingredients_state]
            )

        # ════════════════════════════════════════════════════
        # TAB 2 — Ingredient Substitution
        # ════════════════════════════════════════════════════
        with gr.Tab("🔄 Ingredient Substitution"):
            gr.Markdown("""
            ### 🔄 Find Ingredient Substitutes
            Ask in **natural language** — the system understands your request.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 💬 Ask for a Substitute")
                    sub_query = gr.Textbox(
                        label="Your Substitution Request",
                        placeholder=(
                            "e.g.  'vegan alternative to butter'\n"
                            "      'substitute for paneer'\n"
                            "      'replace ghee'\n"
                            "      'gluten free alternative to atta'"
                        ),
                        lines=4
                    )
                    top_k_slider = gr.Slider(
                        minimum=3, maximum=10, value=5, step=1,
                        label="Number of substitutes to show"
                    )
                    sub_btn = gr.Button(
                        "🔍 Find Best Substitutes",
                        variant="primary",
                        size="lg"
                    )

                    gr.Markdown("""
                    **Try these examples:**
                    - `vegan alternative to butter`
                    - `substitute for paneer`
                    - `replace ghee`
                    - `gluten free alternative to atta`
                    - `instead of toor dal`
                    - `dairy free substitute for curd`
                    """)

                with gr.Column(scale=1):
                    gr.Markdown("### 🌿 Substitution Results")
                    sub_output = gr.Markdown(
                        value="*Substitutes will appear here...*"
                    )

            sub_btn.click(
                fn=run_substitution,
                inputs=[sub_query, top_k_slider],
                outputs=[sub_output]
            )

        # ════════════════════════════════════════════════════
        # TAB 3 — Cost Estimator
        # ════════════════════════════════════════════════════
        with gr.Tab("💸 Cost Estimator"):
            gr.Markdown("""
            ### 💸 Recipe Cost Estimator
            Enter ingredients to get an Indian market price breakdown in ₹.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🧺 Enter Your Ingredients")
                    cost_ing_input = gr.Textbox(
                        label="Ingredients (one per line, quantities optional)",
                        placeholder=(
                            "toor dal\n"
                            "2 onion\n"
                            "3 tomato\n"
                            "ghee\n"
                            "cumin\n"
                            "1/2 tsp turmeric\n"
                            "paneer"
                        ),
                        lines=10
                    )
                    cost_servings = gr.Slider(
                        minimum=1, maximum=10, value=2, step=1,
                        label="👨‍👩‍👧 Number of servings"
                    )
                    cost_btn = gr.Button(
                        "💰 Calculate Cost",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Cost Breakdown")
                    cost_result_output = gr.Markdown(
                        value="*Cost breakdown will appear here...*"
                    )

            cost_btn.click(
                fn=run_cost_estimator,
                inputs=[cost_ing_input, cost_servings],
                outputs=[cost_result_output]
            )

        # ════════════════════════════════════════════════════
        # TAB 4 — Recipe Evaluator
        # ════════════════════════════════════════════════════
        with gr.Tab("📊 Recipe Evaluator"):
            gr.Markdown("""
            ### 📊 Evaluate Any Recipe
            Paste a recipe and constraints to get a quality score using the MCTS reward function.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    eval_steps_input = gr.Textbox(
                        label="🥘 Cooking Steps (one per line)",
                        placeholder=(
                            "Wash and soak toor dal\n"
                            "Pressure cook with turmeric and salt\n"
                            "Heat ghee in pan\n"
                            "Add cumin seeds and let them splutter\n"
                            "Add onion and saute until golden\n"
                            "Add tomatoes and cook until soft\n"
                            "Pour tadka over dal\n"
                            "Garnish with coriander and serve"
                        ),
                        lines=8
                    )
                    eval_ing_input = gr.Textbox(
                        label="📋 Recipe Ingredients (one per line)",
                        placeholder="toor dal\nturmeric\nsalt\nghee\ncumin\nonion\ntomato\ncoriander",
                        lines=5
                    )
                    eval_avail_input = gr.Textbox(
                        label="🧺 Your Available Ingredients (one per line)",
                        placeholder="toor dal\nturmeric\nsalt\ncumin\nonion",
                        lines=5
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Your Constraints")
                    eval_diet = gr.CheckboxGroup(
                        label="🥗 Dietary Restrictions",
                        choices=["vegetarian", "vegan", "jain",
                                 "gluten_free", "diabetic"]
                    )
                    eval_appliances = gr.CheckboxGroup(
                        label="🍳 Available Appliances",
                        choices=["stovetop", "pressure cooker", "microwave",
                                 "oven", "air fryer", "blender"]
                    )
                    with gr.Row():
                        eval_max_time = gr.Slider(
                            minimum=0, maximum=180, value=60, step=5,
                            label="⏱️ Your time budget (mins, 0=no limit)"
                        )
                        eval_est_time = gr.Slider(
                            minimum=0, maximum=180, value=30, step=5,
                            label="🕐 Estimated recipe time (mins)"
                        )

                    eval_btn = gr.Button(
                        "📊 Evaluate Recipe",
                        variant="primary",
                        size="lg"
                    )

                    gr.Markdown("### 🏆 Evaluation Results")
                    eval_result_output = gr.Markdown(
                        value="*Evaluation results will appear here...*"
                    )

            eval_btn.click(
                fn=run_evaluator,
                inputs=[eval_steps_input, eval_ing_input, eval_avail_input,
                        eval_diet, eval_appliances, eval_max_time, eval_est_time],
                outputs=[eval_result_output]
            )

if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft())