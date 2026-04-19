# generation/llm.py — Person 2's file
# Stub so imports don't break. Person 2 fills this in.

def generate_recipe(query: str, retrieved_recipes: list[dict], hf_token: str = None) -> str:
    """
    Generate a recipe using Mistral-7B via HF Inference API.
    TODO: Person 2 implements this.

    Args:
        query: user query
        retrieved_recipes: list of dicts from hybrid_retrieve()
        hf_token: HuggingFace API token

    Returns:
        Generated recipe as a string
    """
    # Temporary template-based fallback used for evaluation
    if not retrieved_recipes:
        return "No recipe found."
    top = retrieved_recipes[0]
    ingredients = ", ".join(top["ingredients"])
    return (
        f"Recipe: {top['title']}\n"
        f"Ingredients: {ingredients}\n"
        f"Instructions: Follow standard cooking procedure."
    )
