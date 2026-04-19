import os
import sys
import re
import importlib.util
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── Load substitution model ────────────────────────────────────
base = os.path.dirname(os.path.abspath(__file__))
sub_path = os.path.join(base, "..", "substitution", "substitution_model.py")
spec = importlib.util.spec_from_file_location("substitution_model", sub_path)
sub_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub_mod)
get_substitutes = sub_mod.get_substitutes

# ── Load embedder for coherence scoring ───────────────────────
embedder = sub_mod.embedder  # reuse already loaded model

# ──────────────────────────────────────────────────────────────
# COMPONENT 1: COHERENCE SCORE
# Checks if cooking steps follow a logical order
# by measuring semantic flow between consecutive steps
# ──────────────────────────────────────────────────────────────

# Expected cooking action order in Indian recipes
COOKING_STAGE_KEYWORDS = [
    # Stage 1 - Prep
    ["wash", "soak", "clean", "peel", "chop", "dice", "slice", "grate",
     "grind", "mince", "marinate", "mix", "prepare", "cut"],
    # Stage 2 - Heat/Base
    ["heat", "warm", "boil", "preheat", "melt"],
    # Stage 3 - Tempering/Frying
    ["add", "saute", "fry", "temper", "splutter", "crackle", "roast",
     "toast", "brown"],
    # Stage 4 - Main cooking
    ["cook", "simmer", "stir", "mix", "combine", "pour", "pressure cook",
     "bake", "steam", "knead"],
    # Stage 5 - Finishing
    ["garnish", "serve", "season", "adjust", "rest", "cool", "plate"]
]

def coherence_score(steps: list[str]) -> float:
    """
    Scores how logically ordered the cooking steps are.
    
    Two components:
    1. Sequential embedding similarity — consecutive steps should
       be semantically related (not jump randomly)
    2. Stage order bonus — steps that follow prep->cook->serve
       order get a bonus
    
    Returns score between 0 and 1.
    """
    if not steps or len(steps) < 2:
        return 0.5  # neutral score if too few steps

    # ── Component 1: Embedding flow ───────────────────────────
    step_embeddings = embedder.encode(steps)
    flow_scores = []
    for i in range(len(step_embeddings) - 1):
        sim = cosine_similarity(
            [step_embeddings[i]],
            [step_embeddings[i + 1]]
        )[0][0]
        flow_scores.append(float(sim))

    avg_flow = np.mean(flow_scores)

    # ── Component 2: Stage order check ────────────────────────
    def get_stage(step_text):
        step_lower = step_text.lower()
        for stage_idx, keywords in enumerate(COOKING_STAGE_KEYWORDS):
            for kw in keywords:
                if kw in step_lower:
                    return stage_idx
        return -1  # unknown stage

    stages = [get_stage(s) for s in steps]
    known_stages = [(i, s) for i, s in enumerate(stages) if s != -1]

    stage_order_score = 0.5  # default neutral
    if len(known_stages) >= 2:
        in_order = sum(
            1 for i in range(len(known_stages) - 1)
            if known_stages[i][1] <= known_stages[i + 1][1]
        )
        stage_order_score = in_order / (len(known_stages) - 1)

    # ── Combined coherence ─────────────────────────────────────
    final = 0.6 * avg_flow + 0.4 * stage_order_score
    return round(float(final), 4)


# ──────────────────────────────────────────────────────────────
# COMPONENT 2: CONSTRAINT SATISFACTION SCORE
# Checks if the recipe respects user constraints
# ──────────────────────────────────────────────────────────────

# Dietary restriction rules
DIETARY_RULES = {
    "vegan": {
        "banned": ["milk", "curd", "yogurt", "paneer", "ghee", "butter",
                   "cream", "khoya", "honey", "egg", "meat", "chicken",
                   "fish", "prawn", "mutton", "condensed milk", "buttermilk"]
    },
    "vegetarian": {
        "banned": ["meat", "chicken", "fish", "prawn", "mutton", "beef",
                   "pork", "egg", "seafood"]
    },
    "jain": {
        "banned": ["onion", "garlic", "potato", "carrot", "radish",
                   "beetroot", "meat", "chicken", "fish", "egg"]
    },
    "gluten_free": {
        "banned": ["wheat", "atta", "maida", "sooji", "bread", "barley",
                   "rye", "semolina"]
    },
    "diabetic": {
        "banned": ["sugar", "jaggery", "honey", "white rice", "maida",
                   "potato", "condensed milk"]
    }
}

# Appliance keywords — what each appliance can do
APPLIANCE_ACTIONS = {
    "pressure cooker": ["pressure cook", "cook", "boil", "steam"],
    "microwave": ["microwave", "heat", "warm", "melt"],
    "oven": ["bake", "roast", "grill", "preheat"],
    "air fryer": ["air fry", "fry", "roast", "crisp"],
    "stovetop": ["fry", "saute", "boil", "simmer", "cook", "temper",
                 "heat", "stir", "roast"],
    "blender": ["blend", "grind", "puree", "mix"],
    "mixer": ["mix", "knead", "blend", "whisk"]
}

def constraint_satisfaction_score(
    recipe_text: str,
    recipe_ingredients: list[str],
    dietary_restrictions: list[str] = None,
    available_appliances: list[str] = None,
    max_time_minutes: int = None,
    estimated_time_minutes: int = None
) -> float:
    """
    Scores how well the recipe satisfies user constraints.
    
    Checks:
    1. Dietary restrictions — no banned ingredients
    2. Appliance compatibility — steps match available appliances
    3. Time constraint — recipe fits within time budget
    
    Returns score between 0 and 1.
    """
    scores = []

    # ── Check 1: Dietary restrictions ─────────────────────────
    if dietary_restrictions:
        for restriction in dietary_restrictions:
            restriction = restriction.lower().strip()
            if restriction in DIETARY_RULES:
                banned = DIETARY_RULES[restriction]["banned"]
                violations = [
                    ing for ing in recipe_ingredients
                    if any(b in ing.lower() for b in banned)
                ]
                diet_score = 1.0 if not violations else max(
                    0.0, 1.0 - (len(violations) * 0.25)
                )
                scores.append(diet_score)

    # ── Check 2: Appliance compatibility ──────────────────────
    if available_appliances:
        recipe_lower = recipe_text.lower()
        # Find all cooking actions mentioned in recipe
        all_actions_in_recipe = []
        for appliance, actions in APPLIANCE_ACTIONS.items():
            for action in actions:
                if action in recipe_lower:
                    all_actions_in_recipe.append((action, appliance))

        # Check if required actions can be done with available appliances
        available_lower = [a.lower() for a in available_appliances]
        compatible = []
        for action, needed_appliance in all_actions_in_recipe:
            can_do = any(
                action in APPLIANCE_ACTIONS.get(avail, [])
                for avail in available_lower
            )
            compatible.append(can_do)

        if compatible:
            appliance_score = sum(compatible) / len(compatible)
            scores.append(appliance_score)

    # ── Check 3: Time constraint ───────────────────────────────
    if max_time_minutes and estimated_time_minutes:
        if estimated_time_minutes <= max_time_minutes:
            time_score = 1.0
        else:
            # Penalize proportionally to how much it exceeds limit
            excess = estimated_time_minutes - max_time_minutes
            time_score = max(0.0, 1.0 - (excess / max_time_minutes))
        scores.append(time_score)

    # ── Return average of all constraint scores ────────────────
    if not scores:
        return 1.0  # no constraints = fully satisfied
    return round(float(np.mean(scores)), 4)


# ──────────────────────────────────────────────────────────────
# COMPONENT 3: INGREDIENT FEASIBILITY SCORE
# Checks if recipe ingredients are available or substitutable
# ──────────────────────────────────────────────────────────────

def ingredient_feasibility_score(
    recipe_ingredients: list[str],
    available_ingredients: list[str]
) -> float:
    """
    Scores how feasible the recipe is given available ingredients.
    
    For each recipe ingredient:
    - If available → full score (1.0)
    - If a substitute exists in available list → partial score (0.6)
    - If neither → zero score (0.0)
    
    Returns score between 0 and 1.
    """
    if not recipe_ingredients:
        return 0.0

    available_lower = [a.lower().strip() for a in available_ingredients]
    ingredient_scores = []

    for ing in recipe_ingredients:
        ing_lower = ing.lower().strip()

        # Check if directly available
        if any(ing_lower in avail or avail in ing_lower
               for avail in available_lower):
            ingredient_scores.append(1.0)
            continue

        # Check if a substitute is available
        try:
            substitutes = get_substitutes(ing_lower, top_k=5)
            sub_names = [s["ingredient"] for s in substitutes]
            has_substitute = any(
                any(sub in avail or avail in sub
                    for avail in available_lower)
                for sub in sub_names
            )
            ingredient_scores.append(0.6 if has_substitute else 0.0)
        except Exception:
            ingredient_scores.append(0.0)

    return round(float(np.mean(ingredient_scores)), 4)


# ──────────────────────────────────────────────────────────────
# FINAL REWARD FUNCTION
# Combines all 3 components into one score
# ──────────────────────────────────────────────────────────────

def compute_reward(
    recipe_steps: list[str],
    recipe_ingredients: list[str],
    available_ingredients: list[str],
    dietary_restrictions: list[str] = None,
    available_appliances: list[str] = None,
    max_time_minutes: int = None,
    estimated_time_minutes: int = None,
    weights: dict = None
) -> dict:
    """
    Master reward function. Computes all 3 scores and returns
    a weighted final reward.

    Default weights:
      coherence            → 0.3
      constraint_satisfaction → 0.4
      ingredient_feasibility  → 0.3

    Returns a dict with all scores for transparency.
    """
    if weights is None:
        weights = {
            "coherence": 0.3,
            "constraint": 0.4,
            "feasibility": 0.3
        }

    c_score = coherence_score(recipe_steps)
    cs_score = constraint_satisfaction_score(
        " ".join(recipe_steps),
        recipe_ingredients,
        dietary_restrictions,
        available_appliances,
        max_time_minutes,
        estimated_time_minutes
    )
    f_score = ingredient_feasibility_score(
        recipe_ingredients,
        available_ingredients
    )

    final_reward = round(
        weights["coherence"] * c_score +
        weights["constraint"] * cs_score +
        weights["feasibility"] * f_score,
        4
    )

    return {
        "coherence_score": c_score,
        "constraint_satisfaction_score": cs_score,
        "ingredient_feasibility_score": f_score,
        "final_reward": final_reward,
        "weights_used": weights
    }


# ── Test ───────────────────────────────────────────────────────
if __name__ == "__main__":

    # Test Recipe: Dal Tadka
    steps = [
        "Wash and soak toor dal for 30 minutes",
        "Pressure cook dal with turmeric and salt for 3 whistles",
        "Heat ghee in a pan",
        "Add cumin seeds and let them splutter",
        "Add chopped onion and saute until golden brown",
        "Add ginger garlic paste and cook for 2 minutes",
        "Add tomatoes and cook until soft",
        "Pour the tadka over cooked dal",
        "Garnish with coriander leaves and serve hot"
    ]

    ingredients = [
        "toor dal", "turmeric", "salt", "ghee", "cumin",
        "onion", "ginger", "garlic", "tomato", "coriander"
    ]

    available = [
        "toor dal", "turmeric", "salt", "ghee", "cumin",
        "onion", "ginger", "garlic", "tomato", "coriander"
    ]

    print("=" * 50)
    print("TEST 1: Dal Tadka — all ingredients available")
    print("=" * 50)
    result = compute_reward(
        recipe_steps=steps,
        recipe_ingredients=ingredients,
        available_ingredients=available,
        dietary_restrictions=["vegetarian"],
        available_appliances=["pressure cooker", "stovetop"],
        max_time_minutes=45,
        estimated_time_minutes=40
    )
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 50)
    print("TEST 2: Dal Tadka — missing some ingredients")
    print("=" * 50)
    partial_available = ["toor dal", "turmeric", "salt", "cumin", "tomato"]
    result2 = compute_reward(
        recipe_steps=steps,
        recipe_ingredients=ingredients,
        available_ingredients=partial_available,
        dietary_restrictions=["vegetarian"],
        available_appliances=["stovetop"],
        max_time_minutes=30,
        estimated_time_minutes=40
    )
    for k, v in result2.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 50)
    print("TEST 3: Vegan constraint violation (has ghee)")
    print("=" * 50)
    result3 = compute_reward(
        recipe_steps=steps,
        recipe_ingredients=ingredients,
        available_ingredients=available,
        dietary_restrictions=["vegan"],
        available_appliances=["pressure cooker", "stovetop"]
    )
    for k, v in result3.items():
        print(f"  {k}: {v}")