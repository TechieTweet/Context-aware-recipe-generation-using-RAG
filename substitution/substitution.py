# substitution/substitutor.py
# Person 3's implementation — wired into the main repo

import os
import sys
import importlib.util

# ── Load person3 substitution model ───────────────────────────
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_sub_path = os.path.join(_base, "substitution", "substitution_model.py")

_spec = importlib.util.spec_from_file_location("substitution_model", _sub_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_get_substitutes = _mod.get_substitutes

# ── Load person3 cost estimator ────────────────────────────────
_cost_path = os.path.join(_base, "cost_estimator", "cost_estimator.py")
_cost_spec = importlib.util.spec_from_file_location("cost_estimator", _cost_path)
_cost_mod = importlib.util.module_from_spec(_cost_spec)
_cost_spec.loader.exec_module(_cost_mod)

_estimate_cost = _cost_mod.estimate_cost

# ── Load person3 reward function ───────────────────────────────
_reward_path = os.path.join(_base, "mcts", "reward_function.py")
_reward_spec = importlib.util.spec_from_file_location("reward_function", _reward_path)
_reward_mod = importlib.util.module_from_spec(_reward_spec)
_reward_spec.loader.exec_module(_reward_mod)

_compute_reward = _reward_mod.compute_reward


def get_substitutes(ingredient: str, top_k: int = 3) -> list[dict]:
    """
    Return ranked ingredient substitutes using FlavorDB + food embeddings.
    Indian ingredients are prioritised.
    """
    try:
        results = _get_substitutes(ingredient, top_k=top_k)
        return results
    except Exception as e:
        return [{"substitute": "N/A", "similarity_score": 0.0, "notes": str(e)}]


def estimate_cost_inr(ingredients: list[str], servings: int = 2) -> dict:
    """
    Estimate total recipe cost in INR with full breakdown.
    """
    try:
        return _estimate_cost(ingredients, servings=servings)
    except Exception as e:
        return {"total_cost": "₹?", "error": str(e)}


def evaluate_recipe(
    recipe_steps: list[str],
    recipe_ingredients: list[str],
    available_ingredients: list[str],
    dietary_restrictions: list[str] = None,
    available_appliances: list[str] = None,
    max_time_minutes: int = None,
    estimated_time_minutes: int = None
) -> dict:
    """
    Evaluate a generated recipe using the MCTS reward function.
    Returns coherence, constraint satisfaction, feasibility scores.
    """
    try:
        return _compute_reward(
            recipe_steps=recipe_steps,
            recipe_ingredients=recipe_ingredients,
            available_ingredients=available_ingredients,
            dietary_restrictions=dietary_restrictions,
            available_appliances=available_appliances,
            max_time_minutes=max_time_minutes,
            estimated_time_minutes=estimated_time_minutes
        )
    except Exception as e:
        return {"error": str(e), "final_reward": 0.0}
