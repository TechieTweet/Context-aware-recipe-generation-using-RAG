import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import importlib.util

# ── Load Indian ingredients list ───────────────────────────────
def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

base = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(base, "..", "data"))

_ing_mod = _load_module("indian_ingredients", os.path.join(data_dir, "indian_ingredients.py"))
INDIAN_INGREDIENTS = _ing_mod.INDIAN_INGREDIENTS

_kb_mod = _load_module("indian_substitutes_kb", os.path.join(data_dir, "indian_substitutes_kb.py"))
INDIAN_SUBSTITUTES_KB = _kb_mod.INDIAN_SUBSTITUTES_KB

# ── Load Dietary Properties ────────────────────────────────────
_diet_mod = _load_module("dietary_properties", os.path.join(data_dir, "dietary_properties.py"))
DIETARY_PROPERTIES = _diet_mod.DIETARY_PROPERTIES
filter_by_dietary_restrictions = _diet_mod.filter_by_dietary_restrictions

# ── Load FlavorDB ──────────────────────────────────────────────
def _resolve_flavordb_path():
    candidates = [
        os.path.join(data_dir, "flavourdb.json"),
        os.path.join(data_dir, "flavordb.json"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find FlavorDB data file. Expected one of: "
        + ", ".join(candidates)
    )


data_path = _resolve_flavordb_path()
with open(data_path, "r", encoding="utf-8") as f:
    flavordb = json.load(f)

flavor_map = {}
for item in flavordb:
    name = item["name"].lower().strip()
    molecules = set(item["flavor_molecules"])
    if molecules:
        flavor_map[name] = molecules

print(f"Loaded {len(flavor_map)} ingredients with flavor data from FlavorDB")

# ── Indian ingredient set for priority boosting ────────────────
INDIAN_SET = set(ing.lower().strip() for ing in INDIAN_INGREDIENTS)

# ── All candidates = FlavorDB + Indian ingredients ─────────────
all_candidates = list(set(
    list(flavor_map.keys()) +
    [ing.lower().strip() for ing in INDIAN_INGREDIENTS]
))

# ── Load Sentence Transformer ──────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Computing embeddings for {len(all_candidates)} ingredients...")
candidate_embeddings = embedder.encode(all_candidates, show_progress_bar=True)
print("Done!")

# ── Flavor Score ───────────────────────────────────────────────
def flavor_score(ing1, ing2):
    set1 = flavor_map.get(ing1.lower().strip(), set())
    set2 = flavor_map.get(ing2.lower().strip(), set())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

# ── Main Function ──────────────────────────────────────────────
def get_substitutes(ingredient, top_k=5, alpha=0.5, indian_boost=0.15, dietary_restrictions=None):
    """
    Strategy:
      1. Check curated KB first — if found, return KB results
         enriched with flavor + embedding scores
      2. If not in KB, fall back to embedding + flavor scoring
      3. Optionally filter by dietary restrictions (vegan, gluten_free, etc.)
    
    Args:
        ingredient (str): The ingredient to find substitutes for
        top_k (int): Number of substitutes to return
        alpha (float): Weight for flavor similarity (1-alpha is embedding weight)
        indian_boost (float): Bonus score for Indian ingredients
        dietary_restrictions (list): List of dietary restrictions to filter by
                                    e.g., ['vegan'], ['gluten_free'], ['vegan', 'gluten_free']
    """
    ingredient = ingredient.lower().strip()

    # ── Strategy 1: Curated KB ─────────────────────────────────
    if ingredient in INDIAN_SUBSTITUTES_KB:
        kb_substitutes = INDIAN_SUBSTITUTES_KB[ingredient]
        
        # Filter by dietary restrictions if provided
        if dietary_restrictions:
            kb_substitutes = filter_by_dietary_restrictions(kb_substitutes, dietary_restrictions)
        
        results = []
        for rank, candidate in enumerate(kb_substitutes):
            candidate = candidate.lower().strip()
            flav_sim = flavor_score(ingredient, candidate)
            # Encode just these candidates for embedding score
            emb1 = embedder.encode([ingredient])
            emb2 = embedder.encode([candidate])
            emb_sim = float(cosine_similarity(emb1, emb2)[0][0])
            base_score = alpha * flav_sim + (1 - alpha) * emb_sim
            is_indian = candidate in INDIAN_SET
            # KB items get a position bonus (first = highest bonus)
            position_bonus = (len(kb_substitutes) - rank) / len(kb_substitutes) * 0.2
            final_score = base_score + position_bonus + (indian_boost if is_indian else 0.0)

            results.append({
                "ingredient": candidate,
                "flavor_similarity": round(flav_sim, 4),
                "embedding_similarity": round(emb_sim, 4),
                "base_score": round(base_score, 4),
                "is_indian": is_indian,
                "source": "curated_kb",
                "final_score": round(final_score, 4)
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]

    # ── Strategy 2: Embedding + Flavor fallback ────────────────
    print(f"'{ingredient}' not in KB, using embedding fallback...")
    query_emb = embedder.encode([ingredient])
    emb_sims = cosine_similarity(query_emb, candidate_embeddings)[0]

    results = []
    for i, candidate in enumerate(all_candidates):
        if candidate == ingredient:
            continue
        
        # Filter by dietary restrictions if provided
        if dietary_restrictions and not filter_by_dietary_restrictions([candidate], dietary_restrictions):
            continue
        
        emb_sim = float(emb_sims[i])
        flav_sim = flavor_score(ingredient, candidate)
        base_score = alpha * flav_sim + (1 - alpha) * emb_sim
        is_indian = candidate in INDIAN_SET
        final_score = base_score + (indian_boost if is_indian else 0.0)

        results.append({
            "ingredient": candidate,
            "flavor_similarity": round(flav_sim, 4),
            "embedding_similarity": round(emb_sim, 4),
            "base_score": round(base_score, 4),
            "is_indian": is_indian,
            "source": "embedding_fallback",
            "final_score": round(final_score, 4)
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:top_k]


# ── Test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    test_ingredients = [
        "cumin", "paneer", "yogurt", "tamarind",
        "ghee", "toor dal", "butter", "atta"
    ]

    print("=" * 70)
    print("STANDARD SUBSTITUTES (No dietary restrictions)")
    print("=" * 70)
    for ing in test_ingredients:
        print(f"\nTop substitutes for '{ing}':")
        subs = get_substitutes(ing, top_k=5)
        for i, s in enumerate(subs, 1):
            tag = "🇮🇳" if s["is_indian"] else "🌍"
            src = "📖" if s["source"] == "curated_kb" else "🤖"
            print(f"  {i}. {tag}{src} {s['ingredient']}")
            print(f"     Flavor: {s['flavor_similarity']} | "
                  f"Embedding: {s['embedding_similarity']} | "
                  f"Final: {s['final_score']}")

    print("\n" + "=" * 70)
    print("VEGAN SUBSTITUTES (dietary_restrictions=['vegan'])")
    print("=" * 70)
    for ing in ["butter", "ghee", "paneer", "yogurt", "milk"]:
        print(f"\nVegan alternatives for '{ing}':")
        subs = get_substitutes(ing, top_k=5, dietary_restrictions=['vegan'])
        if subs:
            for i, s in enumerate(subs, 1):
                tag = "🇮🇳" if s["is_indian"] else "🌍"
                src = "📖" if s["source"] == "curated_kb" else "🤖"
                print(f"  {i}. {tag}{src} {s['ingredient']}")
                print(f"     Flavor: {s['flavor_similarity']} | "
                      f"Embedding: {s['embedding_similarity']} | "
                      f"Final: {s['final_score']}")
        else:
            print("  No vegan alternatives found.")

    print("\n" + "=" * 70)
    print("GLUTEN-FREE SUBSTITUTES (dietary_restrictions=['gluten_free'])")
    print("=" * 70)
    for ing in ["wheat", "atta", "maida", "bread"]:
        print(f"\nGluten-free alternatives for '{ing}':")
        subs = get_substitutes(ing, top_k=5, dietary_restrictions=['gluten_free'])
        if subs:
            for i, s in enumerate(subs, 1):
                tag = "🇮🇳" if s["is_indian"] else "🌍"
                src = "📖" if s["source"] == "curated_kb" else "🤖"
                print(f"  {i}. {tag}{src} {s['ingredient']}")
                print(f"     Flavor: {s['flavor_similarity']} | "
                      f"Embedding: {s['embedding_similarity']} | "
                      f"Final: {s['final_score']}")
        else:
            print("  No gluten-free alternatives found.")

    print("\n" + "=" * 70)
    print("VEGAN + GLUTEN-FREE SUBSTITUTES")
    print("=" * 70)
    for ing in ["butter", "atta"]:
        print(f"\nVegan + Gluten-free alternatives for '{ing}':")
        subs = get_substitutes(ing, top_k=5, dietary_restrictions=['vegan', 'gluten_free'])
        if subs:
            for i, s in enumerate(subs, 1):
                tag = "🇮🇳" if s["is_indian"] else "🌍"
                src = "📖" if s["source"] == "curated_kb" else "🤖"
                print(f"  {i}. {tag}{src} {s['ingredient']}")
                print(f"     Flavor: {s['flavor_similarity']} | "
                      f"Embedding: {s['embedding_similarity']} | "
                      f"Final: {s['final_score']}")
        else:
            print("  No vegan + gluten-free alternatives found.")
