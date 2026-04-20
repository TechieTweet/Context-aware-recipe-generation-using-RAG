# generation/llm.py
# file extended with 3 modes for evaluation comparison
# Mode 1: baseline_generate()  — no RAG, direct LLM call
# Mode 2: naive_rag_generate() — RAG with no re-ranking
# Mode 3: generate_recipe()    — advanced RAG (original function, unchanged)

import os
import requests

# ══════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════

def build_prompt(query: str, retrieved_recipes: list[dict], constraints: dict = None) -> str:
    if constraints is None:
        constraints = {}

    context_blocks = []
    for i, recipe in enumerate(retrieved_recipes, 1):
        ingredients = (
            ", ".join(recipe["ingredients"])
            if isinstance(recipe["ingredients"], list)
            else recipe["ingredients"]
        )
        context_blocks.append(
            f"Reference Recipe {i}: {recipe['title']}\n"
            f"Ingredients: {ingredients}\n"
            f"Instructions: {recipe['full_text'][:500]}"
        )
    context_str = "\n\n".join(context_blocks)

    constraint_lines = []
    if constraints.get("ingredients"):
        ing_list = ", ".join(constraints["ingredients"])
        constraint_lines.append(f"- Available ingredients: {ing_list}. Use ONLY these ingredients unless absolutely essential.")
    if constraints.get("diet"):
        constraint_lines.append(f"- Dietary restriction: {constraints['diet']}. This is mandatory — do not include any ingredients that violate this.")
    if constraints.get("appliance"):
        constraint_lines.append(f"- Appliance constraint: {constraints['appliance']}. Only suggest cooking methods compatible with this.")
    if constraints.get("time"):
        constraint_lines.append(f"- Time limit: {constraints['time']}. The total cooking + prep time must fit within this.")
    if constraints.get("budget"):
        constraint_lines.append(f"- Budget: {constraints['budget']}. Keep the recipe affordable within this limit.")

    constraint_str = (
        "\n".join(constraint_lines)
        if constraint_lines
        else "- No specific constraints. Generate the best recipe possible."
    )

    prompt = f"""You are an expert chef and recipe writer. Your job is to generate a clear, detailed, and practical recipe.

You have been given {len(retrieved_recipes)} reference recipes retrieved from a recipe database. Use them as inspiration and grounding — do not hallucinate ingredients or techniques not supported by these references or the user's available ingredients.

REFERENCE RECIPES ---
{context_str}

USER REQUEST ---
Query: {query}

CONSTRAINTS (you must follow all of these) ---
{constraint_str}

YOUR TASK ---
Generate a complete recipe that:
1. Directly addresses the user's query
2. Respects every constraint listed above
3. Is grounded in the reference recipes — do not invent exotic ingredients
4. Is written in clear, simple steps a home cook can follow

Format your response exactly like this:

Recipe Name: <name>

Ingredients:
- <ingredient 1 with quantity>
- <ingredient 2 with quantity>

Instructions:
1. <step 1>
2. <step 2>

Estimated Time: <prep + cook time>
Serves: <number of servings>

Now generate the recipe:"""

    return prompt
    
# ══════════════════════════════════════════════════════════════
# SHARED GROQ CALLER
# ══════════════════════════════════════════════════════════════

def _call_groq(prompt: str, temperature: float = 0.7, max_tokens: int = 600) -> str:
    """Internal function — calls Groq API with a prompt and returns the text."""
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY not set.")

    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        generated = result["choices"][0]["message"]["content"].strip()
        return generated if generated else "Error: Empty response from model."

    except requests.exceptions.Timeout:
        return "Error: Request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        return f"Error: API returned {e.response.status_code}. {e.response.text[:200]}"
    except Exception as e:
        return f"Error generating recipe: {str(e)}"

# ══════════════════════════════════════════════════════════════
# MODE 1: BASELINE LLM
# No retrieval — raw LLM call with just the query and constraints
# ══════════════════════════════════════════════════════════════

def baseline_generate(query: str, constraints: dict = None) -> str:
    """
    Baseline mode — no RAG. Just sends the query directly to the LLM.
    No retrieved recipes are provided as context.
    Used for evaluation comparison against RAG models.
    """
    if constraints is None:
        constraints = {}

    constraint_lines = []
    if constraints.get("ingredients"):
        ing_list = ", ".join(constraints["ingredients"])
        constraint_lines.append(f"- Available ingredients: {ing_list}")
    if constraints.get("diet"):
        constraint_lines.append(f"- Dietary restriction: {constraints['diet']}")
    if constraints.get("appliance"):
        constraint_lines.append(f"- Appliance: {constraints['appliance']}")
    if constraints.get("time"):
        constraint_lines.append(f"- Time limit: {constraints['time']}")
    if constraints.get("budget"):
        constraint_lines.append(f"- Budget: {constraints['budget']}")

    constraint_str = (
        "\n".join(constraint_lines)
        if constraint_lines
        else "- No specific constraints."
    )

    prompt = f"""You are an expert chef. Generate a complete, practical recipe based on the user's request.

USER REQUEST ---
{query}

CONSTRAINTS ---
{constraint_str}

Format your response exactly like this:

Recipe Name: <name>

Ingredients:
- <ingredient 1 with quantity>
- <ingredient 2 with quantity>

Instructions:
1. <step 1>
2. <step 2>

Estimated Time: <prep + cook time>
Serves: <number of servings>

Now generate the recipe:"""

    return _call_groq(prompt)

# ══════════════════════════════════════════════════════════════
# MODE 2: NAIVE RAG
# Retrieved recipes passed as context but NO re-ranking
# ══════════════════════════════════════════════════════════════

def naive_rag_generate(
    query: str,
    retrieved_recipes: list[dict],
    constraints: dict = None
) -> str:
    """
    Naive RAG mode — uses retrieved recipes as context but applies
    no re-ranking or filtering. Recipes passed in raw retrieval order.
    Used for evaluation comparison against advanced RAG.
    """
    if not retrieved_recipes:
        return "No relevant recipes found."

    context_lines = []
    for i, r in enumerate(retrieved_recipes, 1):
        ingredients = (
            ", ".join(r["ingredients"])
            if isinstance(r["ingredients"], list)
            else r["ingredients"]
        )
        context_lines.append(f"Recipe {i}: {r['title']} | Ingredients: {ingredients}")
    context_str = "\n".join(context_lines)

    constraint_lines = []
    if constraints:
        if constraints.get("diet"):
            constraint_lines.append(f"- Dietary restriction: {constraints['diet']}")
        if constraints.get("appliance"):
            constraint_lines.append(f"- Appliance: {constraints['appliance']}")
        if constraints.get("time"):
            constraint_lines.append(f"- Time limit: {constraints['time']}")

    constraint_str = "\n".join(constraint_lines) if constraint_lines else "- No constraints."

    prompt = f"""You are an expert chef. Use the reference recipes below to generate a recipe.

REFERENCE RECIPES ---
{context_str}

USER REQUEST ---
{query}

CONSTRAINTS ---
{constraint_str}

Format your response exactly like this:

Recipe Name: <name>

Ingredients:
- <ingredient 1 with quantity>
- <ingredient 2 with quantity>

Instructions:
1. <step 1>
2. <step 2>

Estimated Time: <prep + cook time>
Serves: <number of servings>

Now generate the recipe:"""

    return _call_groq(prompt)

# ══════════════════════════════════════════════════════════════
# MODE 3: ADVANCED RAG (Person 2's original function — unchanged)
# Full retrieved context + constraint-aware prompt engineering
# ══════════════════════════════════════════════════════════════

def generate_recipe(
    query: str,
    retrieved_recipes: list[dict],
    hf_token: str = None,
    constraints: dict = None
) -> str:
    """
    Advanced RAG mode — Person 2's original implementation.
    Full retrieved context with detailed constraint-aware prompt.
    This is the primary generation function used in the Gradio app.
    """
    if not retrieved_recipes:
        return "No relevant recipes found. Please try a different query."

    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY not set in .env file.")

    prompt = build_prompt(query, retrieved_recipes, constraints)
    return _call_groq(prompt)
