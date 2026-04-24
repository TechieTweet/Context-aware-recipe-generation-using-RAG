import os
import sys
import importlib.util

# ──────────────────────────────────────────────────────────────
# Load all 3 modules
# ──────────────────────────────────────────────────────────────

base = os.path.dirname(os.path.abspath(__file__))

def _load(name, rel_path):
    path = os.path.join(base, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

print("Loading substitution model...")
sub_mod = _load("substitution_model", "substitution/substitution_model.py")

print("Loading reward function...")
reward_mod = _load("reward_function", "mcts/reward_function.py")

print("Loading cost estimator...")
cost_mod = _load("cost_estimator", "cost_estimator/cost_estimator.py")

print("All modules loaded!\n")

# ──────────────────────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# This is what your teammates will call
# ──────────────────────────────────────────────────────────────

def person3_pipeline(
    recipe_steps: list,
    recipe_ingredients: list,
    available_ingredients: list,
    missing_ingredients: list = None,
    dietary_restrictions: list = None,
    available_appliances: list = None,
    max_time_minutes: int = None,
    estimated_time_minutes: int = None,
    servings: int = 2,
    top_k_substitutes: int = 3
) -> dict:
    """
    Master pipeline for Person 3's components.

    Input:
      recipe_steps          : list of cooking instruction strings
      recipe_ingredients    : list of ingredients the recipe needs
      available_ingredients : list of ingredients user has at home
      missing_ingredients   : ingredients user explicitly says they don't have
                              (if None, auto-detected from available list)
      dietary_restrictions  : e.g. ["vegetarian", "gluten_free"]
      available_appliances  : e.g. ["stovetop", "pressure cooker"]
      max_time_minutes      : user's time budget
      estimated_time_minutes: how long recipe takes
      servings              : number of servings
      top_k_substitutes     : how many substitutes to suggest per ingredient

    Output:
      dict with substitutions, reward score, and cost estimate
    """

    results = {}

    # ── Step 1: Find missing ingredients ──────────────────────
    available_lower = [a.lower().strip() for a in available_ingredients]

    if missing_ingredients is None:
        missing_ingredients = [
            ing for ing in recipe_ingredients
            if not any(
                ing.lower().strip() in avail or avail in ing.lower().strip()
                for avail in available_lower
            )
        ]

    results["missing_ingredients"] = missing_ingredients

    # ── Step 2: Get substitutes for missing ingredients ────────
    substitutions = {}
    for ing in missing_ingredients:
        subs = sub_mod.get_substitutes(ing, top_k=top_k_substitutes)
        substitutions[ing] = subs

    results["substitutions"] = substitutions

    # ── Step 3: Compute reward score ──────────────────────────
    reward = reward_mod.compute_reward(
        recipe_steps=recipe_steps,
        recipe_ingredients=recipe_ingredients,
        available_ingredients=available_ingredients,
        dietary_restrictions=dietary_restrictions,
        available_appliances=available_appliances,
        max_time_minutes=max_time_minutes,
        estimated_time_minutes=estimated_time_minutes
    )
    results["reward"] = reward

    # ── Step 4: Estimate cost ──────────────────────────────────
    cost = cost_mod.estimate_cost(
        recipe_ingredients=recipe_ingredients,
        servings=servings
    )
    results["cost"] = cost

    # ── Step 5: Overall feasibility verdict ───────────────────
    reward_score = reward["final_reward"]
    if reward_score >= 0.75:
        verdict = "✅ Excellent — recipe is highly feasible and well-suited"
    elif reward_score >= 0.55:
        verdict = "⚠️  Good — recipe is feasible with minor adjustments"
    elif reward_score >= 0.35:
        verdict = "🟠 Fair — recipe needs substitutions or constraint changes"
    else:
        verdict = "❌ Poor — recipe is not feasible under current constraints"

    results["verdict"] = verdict
    results["reward_score"] = reward_score

    return results


# ──────────────────────────────────────────────────────────────
# PRETTY PRINT HELPER
# ──────────────────────────────────────────────────────────────

def print_results(results: dict):
    print("\n" + "=" * 55)
    print("PERSON 3 PIPELINE RESULTS")
    print("=" * 55)

    print(f"\n📊 VERDICT: {results['verdict']}")
    print(f"   Reward Score: {results['reward_score']}")

    r = results["reward"]
    print(f"\n🧠 REWARD BREAKDOWN:")
    print(f"   Coherence Score         : {r['coherence_score']}")
    print(f"   Constraint Satisfaction : {r['constraint_satisfaction_score']}")
    print(f"   Ingredient Feasibility  : {r['ingredient_feasibility_score']}")
    print(f"   Final Reward            : {r['final_reward']}")

    print(f"\n💸 COST ESTIMATE:")
    c = results["cost"]
    print(f"   Total Cost    : {c['total_cost']}")
    print(f"   Per Serving   : {c['cost_per_serving']}")
    print(f"   Budget        : {c['budget_category']}")

    if results["missing_ingredients"]:
        print(f"\n🔍 MISSING INGREDIENTS: {results['missing_ingredients']}")
        print(f"\n🔄 SUBSTITUTIONS:")
        for ing, subs in results["substitutions"].items():
            print(f"\n   '{ing}' can be replaced with:")
            for i, s in enumerate(subs, 1):
                tag = "🇮🇳" if s["is_indian"] else "🌍"
                print(f"     {i}. {tag} {s['ingredient']} "
                      f"(score: {s['final_score']})")
    else:
        print(f"\n✅ All ingredients are available — no substitutions needed!")

    print("\n" + "=" * 55)


# ──────────────────────────────────────────────────────────────
# TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Scenario: User wants to make Dal Tadka
    # but is missing ghee and coriander
    # and is vegetarian with only a stovetop

    steps = [
        "Wash and soak toor dal for 30 minutes",
        "Pressure cook dal with turmeric and salt for 3 whistles",
        "Heat oil in a pan",
        "Add cumin seeds and let them splutter",
        "Add chopped onion and saute until golden brown",
        "Add ginger garlic paste and cook for 2 minutes",
        "Add tomatoes and cook until soft",
        "Pour the tadka over cooked dal",
        "Garnish with coriander leaves and serve hot"
    ]

    recipe_ingredients = [
        "toor dal", "turmeric", "salt", "ghee",
        "cumin", "onion", "ginger", "garlic",
        "tomato", "coriander"
    ]

    # User only has these at home
    available_ingredients = [
        "toor dal", "turmeric", "salt",
        "cumin", "onion", "ginger", "garlic", "tomato"
    ]
    # Missing: ghee, coriander

    results = person3_pipeline(
        recipe_steps=steps,
        recipe_ingredients=recipe_ingredients,
        available_ingredients=available_ingredients,
        dietary_restrictions=["vegetarian"],
        available_appliances=["stovetop", "pressure cooker"],
        max_time_minutes=45,
        estimated_time_minutes=40,
        servings=2,
        top_k_substitutes=3
    )

    print_results(results)