from flask import Flask, request, jsonify
import os
import importlib.util

app = Flask(__name__)

# ── Load modules ───────────────────────────────────────────────
base = os.path.dirname(os.path.abspath(__file__))

def _load(name, rel_path):
    path = os.path.join(base, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

print("Loading modules...")
sub_mod = _load("substitution_model", "substitution/substitution_model.py")
reward_mod = _load("reward_function", "mcts/reward_function.py")
cost_mod = _load("cost_estimator", "cost_estimator/cost_estimator.py")
print("All modules loaded. API ready!")

# ══════════════════════════════════════════════════════════════
# ROUTE 1: Health check
# ══════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "module": "Person 3 — Substitution + Reward + Cost",
        "routes": [
            "GET  /",
            "POST /substitute",
            "POST /score",
            "POST /cost",
            "POST /pipeline"
        ]
    })

# ══════════════════════════════════════════════════════════════
# ROUTE 2: Ingredient substitution
# POST /substitute
# ══════════════════════════════════════════════════════════════

@app.route("/substitute", methods=["POST"])
def substitute():
    """
    Input JSON:
    {
        "ingredient": "butter",
        "top_k": 5,           (optional, default 5)
        "indian_boost": 0.15  (optional, default 0.15)
    }

    Output JSON:
    {
        "ingredient": "butter",
        "substitutes": [
            {
                "ingredient": "ghee",
                "flavor_similarity": 0.45,
                "embedding_similarity": 0.72,
                "final_score": 0.73,
                "is_indian": true,
                "source": "curated_kb"
            },
            ...
        ]
    }
    """
    data = request.get_json()

    if not data or "ingredient" not in data:
        return jsonify({"error": "Missing 'ingredient' in request body"}), 400

    ingredient = data["ingredient"]
    top_k = data.get("top_k", 5)
    indian_boost = data.get("indian_boost", 0.15)

    try:
        subs = sub_mod.get_substitutes(
            ingredient,
            top_k=top_k,
            indian_boost=indian_boost
        )
        return jsonify({
            "ingredient": ingredient,
            "substitutes": subs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════
# ROUTE 3: Recipe scoring
# POST /score
# ══════════════════════════════════════════════════════════════

@app.route("/score", methods=["POST"])
def score():
    """
    Input JSON:
    {
        "recipe_steps": ["Wash dal", "Pressure cook", ...],
        "recipe_ingredients": ["toor dal", "ghee", ...],
        "available_ingredients": ["toor dal", "cumin", ...],
        "dietary_restrictions": ["vegetarian"],     (optional)
        "available_appliances": ["stovetop"],        (optional)
        "max_time_minutes": 45,                      (optional)
        "estimated_time_minutes": 40                 (optional)
    }

    Output JSON:
    {
        "coherence_score": 0.72,
        "constraint_satisfaction_score": 1.0,
        "ingredient_feasibility_score": 0.85,
        "final_reward": 0.84,
        "weights_used": {...}
    }
    """
    data = request.get_json()

    required = ["recipe_steps", "recipe_ingredients", "available_ingredients"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing '{field}' in request body"}), 400

    try:
        result = reward_mod.compute_reward(
            recipe_steps=data["recipe_steps"],
            recipe_ingredients=data["recipe_ingredients"],
            available_ingredients=data["available_ingredients"],
            dietary_restrictions=data.get("dietary_restrictions"),
            available_appliances=data.get("available_appliances"),
            max_time_minutes=data.get("max_time_minutes"),
            estimated_time_minutes=data.get("estimated_time_minutes")
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════
# ROUTE 4: Cost estimation
# POST /cost
# ══════════════════════════════════════════════════════════════

@app.route("/cost", methods=["POST"])
def cost():
    """
    Input JSON:
    {
        "ingredients": ["toor dal", "2 onion", "ghee", ...],
        "servings": 2   (optional, default 2)
    }

    Output JSON:
    {
        "total_cost": "₹85",
        "cost_per_serving": "₹42.5",
        "servings": 2,
        "budget_category": "💚 Budget friendly",
        "breakdown": [...],
        "ingredients_not_found": [...]
    }
    """
    data = request.get_json()

    if not data or "ingredients" not in data:
        return jsonify({"error": "Missing 'ingredients' in request body"}), 400

    try:
        result = cost_mod.estimate_cost(
            recipe_ingredients=data["ingredients"],
            servings=data.get("servings", 2)
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════
# ROUTE 5: Full pipeline
# POST /pipeline
# ══════════════════════════════════════════════════════════════

@app.route("/pipeline", methods=["POST"])
def pipeline():
    """
    Main route — Person 2 calls this after generating a recipe.

    Input JSON:
    {
        "recipe_steps": ["Wash dal", "Heat oil", ...],
        "recipe_ingredients": ["toor dal", "ghee", "cumin", ...],
        "available_ingredients": ["toor dal", "cumin", ...],
        "dietary_restrictions": ["vegetarian"],   (optional)
        "available_appliances": ["stovetop"],      (optional)
        "max_time_minutes": 45,                    (optional)
        "estimated_time_minutes": 40,              (optional)
        "servings": 2,                             (optional)
        "top_k_substitutes": 3                     (optional)
    }

    Output JSON:
    {
        "missing_ingredients": ["ghee"],
        "substitutions": {
            "ghee": [
                {"ingredient": "butter", "final_score": 0.73, ...},
                ...
            ]
        },
        "reward": {
            "coherence_score": 0.72,
            "constraint_satisfaction_score": 1.0,
            "ingredient_feasibility_score": 0.85,
            "final_reward": 0.84
        },
        "cost": {
            "total_cost": "₹85",
            "cost_per_serving": "₹42.5",
            "budget_category": "💚 Budget friendly",
            ...
        },
        "verdict": "✅ Excellent — recipe is highly feasible",
        "reward_score": 0.84
    }
    """
    data = request.get_json()

    required = ["recipe_steps", "recipe_ingredients", "available_ingredients"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing '{field}' in request body"}), 400

    try:
        recipe_steps = data["recipe_steps"]
        recipe_ingredients = data["recipe_ingredients"]
        available_ingredients = data["available_ingredients"]
        dietary_restrictions = data.get("dietary_restrictions")
        available_appliances = data.get("available_appliances")
        max_time = data.get("max_time_minutes")
        est_time = data.get("estimated_time_minutes")
        servings = data.get("servings", 2)
        top_k = data.get("top_k_substitutes", 3)

        # Find missing ingredients
        available_lower = [a.lower().strip() for a in available_ingredients]
        missing = [
            ing for ing in recipe_ingredients
            if not any(
                ing.lower().strip() in avail or avail in ing.lower().strip()
                for avail in available_lower
            )
        ]

        # Get substitutions
        substitutions = {}
        for ing in missing:
            substitutions[ing] = sub_mod.get_substitutes(ing, top_k=top_k)

        # Compute reward
        reward = reward_mod.compute_reward(
            recipe_steps=recipe_steps,
            recipe_ingredients=recipe_ingredients,
            available_ingredients=available_ingredients,
            dietary_restrictions=dietary_restrictions,
            available_appliances=available_appliances,
            max_time_minutes=max_time,
            estimated_time_minutes=est_time
        )

        # Estimate cost
        cost = cost_mod.estimate_cost(recipe_ingredients, servings=servings)

        # Verdict
        reward_score = reward["final_reward"]
        if reward_score >= 0.75:
            verdict = "✅ Excellent — recipe is highly feasible and well-suited"
        elif reward_score >= 0.55:
            verdict = "⚠️ Good — recipe is feasible with minor adjustments"
        elif reward_score >= 0.35:
            verdict = "🟠 Fair — recipe needs substitutions or constraint changes"
        else:
            verdict = "❌ Poor — recipe is not feasible under current constraints"

        return jsonify({
            "missing_ingredients": missing,
            "substitutions": substitutions,
            "reward": reward,
            "cost": cost,
            "verdict": verdict,
            "reward_score": reward_score
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5001)