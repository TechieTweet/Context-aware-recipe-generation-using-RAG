# Dietary and allergen properties for ingredients
# Maps ingredient names to their dietary attributes

DIETARY_PROPERTIES = {
    # Spices (mostly vegan, vegetarian, gluten-free)
    "cumin": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "coriander": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "turmeric": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "ginger": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "garlic": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "mustard seeds": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "fenugreek": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "cardamom": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "clove": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "cinnamon": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "black pepper": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "red chili": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "green chili": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "asafoetida": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "saffron": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "fennel seeds": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "star anise": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "nutmeg": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "mace": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "poppy seeds": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "bay leaf": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "curry leaves": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "carom seeds": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "dry mango powder": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "chaat masala": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "garam masala": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "kasuri methi": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "black cardamom": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "stone flower": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "dried red chili": {"vegan": True, "vegetarian": True, "gluten_free": True},

    # Vegetables (all vegan, vegetarian, gluten-free)
    "onion": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "tomato": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "potato": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "spinach": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "okra": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "eggplant": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "brinjal": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "bitter gourd": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "bottle gourd": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "ridge gourd": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "snake gourd": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "drumstick": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "raw banana": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "raw papaya": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "yam": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "taro root": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "lotus stem": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "cauliflower": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "cabbage": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "peas": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "carrot": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "radish": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "beetroot": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "french beans": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "cluster beans": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "colocasia": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "raw jackfruit": {"vegan": True, "vegetarian": True, "gluten_free": True},

    # Lentils and Legumes (vegan, vegetarian, gluten-free)
    "toor dal": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "moong dal": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "chana dal": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "urad dal": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "masoor dal": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "rajma": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "chickpea": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "black eyed peas": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "horse gram": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "green moong": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "lobia": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "soybean": {"vegan": True, "vegetarian": True, "gluten_free": True},

    # Grains and Flours
    "rice": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "wheat": {"vegan": True, "vegetarian": True, "gluten_free": False},
    "atta": {"vegan": True, "vegetarian": True, "gluten_free": False},
    "maida": {"vegan": True, "vegetarian": True, "gluten_free": False},
    "besan": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "sooji": {"vegan": True, "vegetarian": True, "gluten_free": False},
    "poha": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "ragi": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "bajra": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "jowar": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "barley": {"vegan": True, "vegetarian": True, "gluten_free": False},
    "sabudana": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "bread": {"vegan": False, "vegetarian": True, "gluten_free": False},

    # Dairy
    "milk": {"vegan": False, "vegetarian": True, "gluten_free": True},
    "curd": {"vegan": False, "vegetarian": True, "gluten_free": True},
    "yogurt": {"vegan": False, "vegetarian": True, "gluten_free": True},
    "paneer": {"vegan": False, "vegetarian": True, "gluten_free": True},
    "ghee": {"vegan": False, "vegetarian": True, "gluten_free": True},
    "butter": {"vegan": False, "vegetarian": True, "gluten_free": True},
    "cream": {"vegan": False, "vegetarian": True, "gluten_free": True},
    "khoya": {"vegan": False, "vegetarian": True, "gluten_free": True},
    "condensed milk": {"vegan": False, "vegetarian": True, "gluten_free": True},
    "buttermilk": {"vegan": False, "vegetarian": True, "gluten_free": True},

    # Oils and Fats (vegan, vegetarian, gluten-free)
    "coconut oil": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "mustard oil": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "groundnut oil": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "sesame oil": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "sunflower oil": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "refined oil": {"vegan": True, "vegetarian": True, "gluten_free": True},

    # Nuts and Seeds (vegan, vegetarian, gluten-free)
    "cashew": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "almond": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "peanut": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "sesame seeds": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "coconut": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "watermelon seeds": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "pumpkin seeds": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "flax seeds": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "walnut": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "pistachio": {"vegan": True, "vegetarian": True, "gluten_free": True},

    # Fruits (vegan, vegetarian, gluten-free)
    "mango": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "banana": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "tamarind": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "lemon": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "lime": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "amla": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "raw mango": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "dates": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "raisins": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "pomegranate": {"vegan": True, "vegetarian": True, "gluten_free": True},

    # Sweeteners (vegan, vegetarian, gluten-free)
    "sugar": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "jaggery": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "honey": {"vegan": False, "vegetarian": True, "gluten_free": True},
    "mishri": {"vegan": True, "vegetarian": True, "gluten_free": True},

    # Others
    "salt": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "vinegar": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "rose water": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "kewra water": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "baking soda": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "baking powder": {"vegan": True, "vegetarian": True, "gluten_free": True},
    "corn flour": {"vegan": True, "vegetarian": True, "gluten_free": True},
}


def get_dietary_status(ingredient, dietary_type):
    """
    Check if an ingredient matches a dietary requirement.
    
    Args:
        ingredient (str): The ingredient name
        dietary_type (str): One of 'vegan', 'vegetarian', 'gluten_free'
    
    Returns:
        bool: True if ingredient matches the dietary requirement
    """
    ingredient = ingredient.lower().strip()
    dietary_type = dietary_type.lower().strip().replace("-", "_")
    
    if ingredient not in DIETARY_PROPERTIES:
        # Default: assume ingredient is available if not in database
        return True
    
    properties = DIETARY_PROPERTIES[ingredient]
    return properties.get(dietary_type, True)


def filter_by_dietary_restrictions(candidates, restrictions):
    """
    Filter a list of ingredient candidates by dietary restrictions.
    
    Args:
        candidates (list): List of ingredient names
        restrictions (list): List of dietary restrictions (e.g., ['vegan', 'gluten_free'])
    
    Returns:
        list: Filtered list of candidates matching ALL restrictions
    """
    if not restrictions:
        return candidates
    
    filtered = []
    for candidate in candidates:
        matches_all = True
        for restriction in restrictions:
            if not get_dietary_status(candidate, restriction):
                matches_all = False
                break
        if matches_all:
            filtered.append(candidate)
    
    return filtered
