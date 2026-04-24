import os
import re

# ──────────────────────────────────────────────────────────────
# INDIAN MARKET PRICES DATABASE
# Prices in ₹ per standard unit (approximate 2024 prices)
# Sources: BigBasket, Blinkit, local market averages
# ──────────────────────────────────────────────────────────────

PRICE_DB = {
    # Lentils and Legumes (per 100g)
    "toor dal":         {"price": 14, "unit": "100g"},
    "moong dal":        {"price": 12, "unit": "100g"},
    "chana dal":        {"price": 10, "unit": "100g"},
    "urad dal":         {"price": 13, "unit": "100g"},
    "masoor dal":       {"price": 10, "unit": "100g"},
    "rajma":            {"price": 14, "unit": "100g"},
    "chickpea":         {"price": 10, "unit": "100g"},
    "green moong":      {"price": 12, "unit": "100g"},
    "black eyed peas":  {"price": 10, "unit": "100g"},
    "horse gram":       {"price": 8,  "unit": "100g"},
    "lobia":            {"price": 10, "unit": "100g"},
    "soybean":          {"price": 8,  "unit": "100g"},

    # Grains and Flours (per 100g)
    "rice":             {"price": 6,  "unit": "100g"},
    "wheat":            {"price": 4,  "unit": "100g"},
    "atta":             {"price": 5,  "unit": "100g"},
    "maida":            {"price": 4,  "unit": "100g"},
    "besan":            {"price": 8,  "unit": "100g"},
    "sooji":            {"price": 5,  "unit": "100g"},
    "poha":             {"price": 6,  "unit": "100g"},
    "ragi":             {"price": 7,  "unit": "100g"},
    "bajra":            {"price": 5,  "unit": "100g"},
    "jowar":            {"price": 5,  "unit": "100g"},
    "barley":           {"price": 6,  "unit": "100g"},
    "sabudana":         {"price": 10, "unit": "100g"},
    "corn flour":       {"price": 6,  "unit": "100g"},

    # Dairy (per standard unit)
    "milk":             {"price": 28, "unit": "500ml"},
    "curd":             {"price": 30, "unit": "500g"},
    "yogurt":           {"price": 30, "unit": "500g"},
    "paneer":           {"price": 80, "unit": "200g"},
    "ghee":             {"price": 55, "unit": "100g"},
    "butter":           {"price": 55, "unit": "100g"},
    "cream":            {"price": 35, "unit": "200ml"},
    "khoya":            {"price": 50, "unit": "100g"},
    "condensed milk":   {"price": 60, "unit": "400g"},
    "buttermilk":       {"price": 20, "unit": "500ml"},

    # Vegetables (per piece or 100g)
    "onion":            {"price": 3,  "unit": "piece"},
    "tomato":           {"price": 4,  "unit": "piece"},
    "potato":           {"price": 3,  "unit": "piece"},
    "spinach":          {"price": 20, "unit": "bunch"},
    "okra":             {"price": 8,  "unit": "100g"},
    "eggplant":         {"price": 15, "unit": "piece"},
    "brinjal":          {"price": 15, "unit": "piece"},
    "bitter gourd":     {"price": 10, "unit": "piece"},
    "bottle gourd":     {"price": 20, "unit": "piece"},
    "cauliflower":      {"price": 30, "unit": "piece"},
    "cabbage":          {"price": 25, "unit": "piece"},
    "peas":             {"price": 15, "unit": "100g"},
    "carrot":           {"price": 5,  "unit": "piece"},
    "radish":           {"price": 5,  "unit": "piece"},
    "beetroot":         {"price": 8,  "unit": "piece"},
    "french beans":     {"price": 12, "unit": "100g"},
    "raw banana":       {"price": 8,  "unit": "piece"},
    "yam":              {"price": 10, "unit": "100g"},
    "taro root":        {"price": 10, "unit": "100g"},
    "raw jackfruit":    {"price": 30, "unit": "piece"},
    "drumstick":        {"price": 5,  "unit": "piece"},

    # Spices (per small quantity used in cooking)
    "cumin":            {"price": 2,  "unit": "tsp"},
    "coriander":        {"price": 1,  "unit": "tsp"},
    "turmeric":         {"price": 1,  "unit": "tsp"},
    "ginger":           {"price": 3,  "unit": "piece"},
    "garlic":           {"price": 2,  "unit": "piece"},
    "mustard seeds":    {"price": 1,  "unit": "tsp"},
    "fenugreek":        {"price": 1,  "unit": "tsp"},
    "cardamom":         {"price": 3,  "unit": "piece"},
    "clove":            {"price": 2,  "unit": "piece"},
    "cinnamon":         {"price": 2,  "unit": "piece"},
    "black pepper":     {"price": 1,  "unit": "tsp"},
    "red chili":        {"price": 1,  "unit": "tsp"},
    "green chili":      {"price": 1,  "unit": "piece"},
    "asafoetida":       {"price": 1,  "unit": "pinch"},
    "saffron":          {"price": 10, "unit": "pinch"},
    "fennel seeds":     {"price": 1,  "unit": "tsp"},
    "star anise":       {"price": 2,  "unit": "piece"},
    "nutmeg":           {"price": 2,  "unit": "piece"},
    "bay leaf":         {"price": 1,  "unit": "piece"},
    "curry leaves":     {"price": 2,  "unit": "sprig"},
    "carom seeds":      {"price": 1,  "unit": "tsp"},
    "dry mango powder": {"price": 1,  "unit": "tsp"},
    "garam masala":     {"price": 2,  "unit": "tsp"},
    "kasuri methi":     {"price": 1,  "unit": "tsp"},
    "chaat masala":     {"price": 2,  "unit": "tsp"},
    "tamarind":         {"price": 5,  "unit": "small ball"},

    # Oils (per tbsp)
    "coconut oil":      {"price": 5,  "unit": "tbsp"},
    "mustard oil":      {"price": 4,  "unit": "tbsp"},
    "groundnut oil":    {"price": 4,  "unit": "tbsp"},
    "sesame oil":       {"price": 5,  "unit": "tbsp"},
    "sunflower oil":    {"price": 3,  "unit": "tbsp"},
    "refined oil":      {"price": 3,  "unit": "tbsp"},
    "oil":              {"price": 3,  "unit": "tbsp"},

    # Nuts and Seeds (per 10g)
    "cashew":           {"price": 15, "unit": "10g"},
    "almond":           {"price": 12, "unit": "10g"},
    "peanut":           {"price": 4,  "unit": "10g"},
    "sesame seeds":     {"price": 3,  "unit": "tsp"},
    "coconut":          {"price": 20, "unit": "piece"},
    "poppy seeds":      {"price": 3,  "unit": "tsp"},
    "walnut":           {"price": 15, "unit": "10g"},
    "pistachio":        {"price": 20, "unit": "10g"},

    # Fruits
    "mango":            {"price": 30, "unit": "piece"},
    "banana":           {"price": 5,  "unit": "piece"},
    "lemon":            {"price": 5,  "unit": "piece"},
    "lime":             {"price": 5,  "unit": "piece"},
    "tamarind":         {"price": 5,  "unit": "small ball"},
    "pomegranate":      {"price": 40, "unit": "piece"},
    "dates":            {"price": 5,  "unit": "piece"},

    # Sweeteners
    "sugar":            {"price": 5,  "unit": "100g"},
    "jaggery":          {"price": 6,  "unit": "100g"},
    "honey":            {"price": 15, "unit": "tbsp"},

    # Others
    "salt":             {"price": 1,  "unit": "tsp"},
    "baking soda":      {"price": 1,  "unit": "tsp"},
    "baking powder":    {"price": 1,  "unit": "tsp"},
    "vinegar":          {"price": 2,  "unit": "tbsp"},
    "rose water":       {"price": 3,  "unit": "tbsp"},
    "bread":            {"price": 5,  "unit": "slice"},
    "water":            {"price": 0,  "unit": "cup"},
}

# ──────────────────────────────────────────────────────────────
# HELPER: Fuzzy match ingredient name to price DB
# ──────────────────────────────────────────────────────────────

def find_price(ingredient_name: str):
    """
    Looks up ingredient in price DB.
    Tries exact match first, then partial match.
    Returns (price, unit) or None if not found.
    """
    ing = ingredient_name.lower().strip()

    # Exact match
    if ing in PRICE_DB:
        return PRICE_DB[ing]["price"], PRICE_DB[ing]["unit"]

    # Partial match — check if any key is contained in the ingredient name
    for key in PRICE_DB:
        if key in ing or ing in key:
            return PRICE_DB[key]["price"], PRICE_DB[key]["unit"]

    return None, None


# ──────────────────────────────────────────────────────────────
# MAIN FUNCTION: Estimate recipe cost
# ──────────────────────────────────────────────────────────────

def estimate_cost(
    recipe_ingredients: list,
    servings: int = 2
) -> dict:
    """
    Estimates the cost of a recipe in Indian Rupees.

    Parameters:
      recipe_ingredients : list of ingredient strings
                           Can be plain strings like "cumin"
                           or with quantity like "2 tbsp ghee"
      servings           : number of servings (default 2)

    Returns:
      dict with per-ingredient breakdown and total cost
    """
    breakdown = []
    total_cost = 0
    not_found = []

    for item in recipe_ingredients:
        item_clean = item.lower().strip()

        # Try to extract quantity multiplier from ingredient string
        # e.g. "2 onions" → multiplier 2, "1/2 tsp cumin" → multiplier 0.5
        multiplier = 1.0
        qty_match = re.match(r"^(\d+\.?\d*)\s+", item_clean)
        if qty_match:
            multiplier = float(qty_match.group(1))
            item_clean = item_clean[qty_match.end():]  # remove quantity

        # Also handle fractions like "1/2"
        frac_match = re.match(r"^(\d+)/(\d+)\s+", item_clean)
        if frac_match:
            multiplier = float(frac_match.group(1)) / float(frac_match.group(2))
            item_clean = item_clean[frac_match.end():]

        price, unit = find_price(item_clean)

        if price is not None:
            item_cost = round(price * multiplier, 2)
            total_cost += item_cost
            breakdown.append({
                "ingredient": item,
                "matched_to": item_clean,
                "unit_price": f"₹{price}/{unit}",
                "multiplier": multiplier,
                "cost": f"₹{item_cost}"
            })
        else:
            not_found.append(item)
            breakdown.append({
                "ingredient": item,
                "matched_to": None,
                "unit_price": "unknown",
                "multiplier": multiplier,
                "cost": "unknown"
            })

    cost_per_serving = round(total_cost / servings, 2)

    return {
        "total_cost": f"₹{round(total_cost, 2)}",
        "cost_per_serving": f"₹{cost_per_serving}",
        "servings": servings,
        "breakdown": breakdown,
        "ingredients_not_found": not_found,
        "budget_category": classify_budget(total_cost)
    }


def classify_budget(total_cost: float) -> str:
    """
    Classifies recipe into budget categories
    relevant to Indian households.
    """
    if total_cost <= 50:
        return "💚 Budget friendly (under ₹50)"
    elif total_cost <= 150:
        return "💛 Moderate (₹50–₹150)"
    elif total_cost <= 300:
        return "🟠 Slightly expensive (₹150–₹300)"
    else:
        return "🔴 Premium (above ₹300)"


# ── Test ───────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 50)
    print("TEST 1: Dal Tadka")
    print("=" * 50)
    dal_tadka = [
        "toor dal", "turmeric", "salt", "ghee",
        "cumin", "onion", "ginger", "garlic",
        "tomato", "coriander", "green chili"
    ]
    result = estimate_cost(dal_tadka, servings=2)
    print(f"Total cost    : {result['total_cost']}")
    print(f"Per serving   : {result['cost_per_serving']}")
    print(f"Budget        : {result['budget_category']}")
    print(f"\nBreakdown:")
    for item in result["breakdown"]:
        print(f"  {item['ingredient']:20} → {item['cost']:10} ({item['unit_price']})")
    if result["ingredients_not_found"]:
        print(f"\nNot found in DB: {result['ingredients_not_found']}")

    print("\n" + "=" * 50)
    print("TEST 2: Paneer Butter Masala")
    print("=" * 50)
    pbm = [
        "paneer", "butter", "cream", "tomato",
        "onion", "cashew", "ginger", "garlic",
        "garam masala", "red chili", "salt", "sugar"
    ]
    result2 = estimate_cost(pbm, servings=3)
    print(f"Total cost    : {result2['total_cost']}")
    print(f"Per serving   : {result2['cost_per_serving']}")
    print(f"Budget        : {result2['budget_category']}")
    print(f"\nBreakdown:")
    for item in result2["breakdown"]:
        print(f"  {item['ingredient']:20} → {item['cost']:10} ({item['unit_price']})")

    print("\n" + "=" * 50)
    print("TEST 3: With quantities")
    print("=" * 50)
    with_qty = [
        "2 onion", "3 tomato", "1/2 tsp turmeric",
        "2 tbsp ghee", "100g paneer", "1 tsp cumin"
    ]
    result3 = estimate_cost(with_qty, servings=2)
    print(f"Total cost    : {result3['total_cost']}")
    print(f"Per serving   : {result3['cost_per_serving']}")
    print(f"Budget        : {result3['budget_category']}")
    print(f"\nBreakdown:")
    for item in result3["breakdown"]:
        print(f"  {item['ingredient']:20} → {item['cost']:10} ({item['unit_price']})")