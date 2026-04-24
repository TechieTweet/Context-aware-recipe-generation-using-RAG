# Curated Indian ingredient substitution knowledge base
# Based on culinary knowledge — what actually works in Indian cooking
# Format: "ingredient" -> list of substitutes in order of preference

INDIAN_SUBSTITUTES_KB = {
    # Dairy
    "ghee": ["butter", "coconut oil", "refined oil", "mustard oil"],
    "paneer": ["tofu", "cottage cheese", "curd", "khoya"],
    "curd": ["yogurt", "buttermilk", "sour cream", "coconut milk"],
    "yogurt": ["curd", "buttermilk", "coconut milk", "milk"],
    "buttermilk": ["curd", "yogurt", "milk", "coconut milk"],
    "butter": ["ghee", "coconut oil", "refined oil", "groundnut oil"],
    "cream": ["coconut milk", "cashew cream", "milk", "curd"],
    "khoya": ["condensed milk", "milk powder", "paneer"],
    "milk": ["coconut milk", "almond milk", "curd", "buttermilk"],

    # Spices
    "salt": ["rock salt", "sea salt", "sendha namak"],
    "cumin": ["caraway seeds", "fennel seeds", "coriander", "carom seeds"],
    "coriander": ["cumin", "parsley", "curry leaves", "fennel seeds"],
    "turmeric": ["saffron", "paprika", "annatto", "ginger"],
    "ginger": ["dry ginger powder", "galangal", "turmeric", "garlic"],
    "garlic": ["asafoetida", "ginger", "onion powder", "garlic powder"],
    "cardamom": ["cinnamon", "nutmeg", "allspice", "clove"],
    "clove": ["cardamom", "allspice", "cinnamon", "nutmeg"],
    "cinnamon": ["cardamom", "clove", "allspice", "star anise"],
    "star anise": ["fennel seeds", "anise seeds", "clove", "cinnamon"],
    "fennel seeds": ["star anise", "cumin", "caraway seeds", "anise seeds"],
    "mustard seeds": ["cumin", "carom seeds", "fennel seeds", "celery seeds"],
    "fenugreek": ["kasuri methi", "mustard seeds", "celery seeds", "fennel seeds"],
    "kasuri methi": ["fenugreek", "dried basil", "dried oregano"],
    "asafoetida": ["garlic", "onion powder", "garlic powder"],
    "carom seeds": ["cumin", "thyme", "fennel seeds", "oregano"],
    "saffron": ["turmeric", "marigold petals", "paprika"],
    "black pepper": ["white pepper", "red chili", "cayenne pepper"],
    "red chili": ["black pepper", "paprika", "cayenne pepper", "green chili"],
    "green chili": ["red chili", "black pepper", "cayenne pepper"],
    "dry mango powder": ["tamarind", "lemon juice", "lime juice", "vinegar"],
    "tamarind": ["dry mango powder", "lemon juice", "lime juice", "vinegar"],
    "garam masala": ["chole masala", "curry powder", "allspice"],
    "curry leaves": ["bay leaf", "kaffir lime leaves", "basil"],
    "bay leaf": ["curry leaves", "thyme", "oregano"],
    "nutmeg": ["mace", "allspice", "cardamom", "cinnamon"],
    "mace": ["nutmeg", "allspice", "cardamom"],

    # Lentils and Legumes
    "toor dal": ["moong dal", "masoor dal", "chana dal", "yellow split peas"],
    "moong dal": ["toor dal", "masoor dal", "chana dal", "green moong"],
    "chana dal": ["toor dal", "moong dal", "yellow split peas", "chickpea"],
    "urad dal": ["moong dal", "chana dal", "black eyed peas", "lobia"],
    "masoor dal": ["toor dal", "moong dal", "chana dal", "red lentils"],
    "rajma": ["chickpea", "black eyed peas", "kidney beans", "lobia"],
    "chickpea": ["rajma", "black eyed peas", "chana dal", "lobia"],
    "green moong": ["moong dal", "toor dal", "peas", "edamame"],

    # Grains and Flours
    "atta": ["maida", "whole wheat flour", "spelt flour"],
    "maida": ["atta", "all purpose flour", "corn flour"],
    "besan": ["chickpea flour", "soy flour", "rice flour", "corn flour"],
    "sooji": ["rice flour", "corn flour", "besan", "maida"],
    "rice": ["quinoa", "barley", "millets", "cauliflower rice"],
    "poha": ["sooji", "rice", "oats", "bread"],
    "sabudana": ["tapioca", "rice", "sooji"],
    "ragi": ["bajra", "jowar", "besan", "atta"],
    "bajra": ["ragi", "jowar", "atta", "besan"],
    "jowar": ["bajra", "ragi", "atta", "corn flour"],

    # Vegetables
    "onion": ["shallots", "spring onion", "leek", "asafoetida"],
    "tomato": ["tamarind", "dry mango powder", "lemon juice", "canned tomatoes"],
    "potato": ["sweet potato", "yam", "taro root", "raw banana"],
    "spinach": ["fenugreek", "amaranth leaves", "kale", "cabbage"],
    "okra": ["green beans", "zucchini", "asparagus"],
    "eggplant": ["zucchini", "mushroom", "raw jackfruit", "potato"],
    "bitter gourd": ["zucchini", "green beans", "ridge gourd"],
    "cauliflower": ["broccoli", "cabbage", "raw banana", "potato"],
    "peas": ["edamame", "corn", "green moong", "french beans"],
    "raw jackfruit": ["eggplant", "mushroom", "banana blossom", "tofu"],
    "drumstick": ["green beans", "asparagus", "french beans"],

    # Oils
    "coconut oil": ["ghee", "refined oil", "butter", "sesame oil"],
    "mustard oil": ["sesame oil", "groundnut oil", "refined oil"],
    "groundnut oil": ["refined oil", "sunflower oil", "coconut oil"],
    "sesame oil": ["groundnut oil", "mustard oil", "coconut oil"],
    "oil":  ["ghee", "butter", "coconut oil", "refined oil"],

    # Nuts and Seeds
    "cashew": ["almond", "peanut", "pistachio", "sunflower seeds"],
    "almond": ["cashew", "walnut", "pistachio", "peanut"],
    "peanut": ["cashew", "almond", "sesame seeds", "sunflower seeds"],
    "coconut": ["cashew", "almond", "oats", "desiccated coconut"],
    "sesame seeds": ["poppy seeds", "flax seeds", "sunflower seeds"],
    "poppy seeds": ["sesame seeds", "flax seeds", "sunflower seeds"],

    # Sweeteners
    "jaggery": ["brown sugar", "sugar", "honey", "maple syrup"],
    "sugar": ["jaggery", "honey", "coconut sugar", "dates"],
    "honey": ["jaggery", "sugar", "maple syrup", "agave"],

    # Souring agents
    "lemon": ["lime", "tamarind", "dry mango powder", "vinegar"],
    "lime": ["lemon", "tamarind", "dry mango powder", "vinegar"],
    "vinegar": ["lemon juice", "lime juice", "tamarind", "dry mango powder"],

    # Others
    "tapioca": ["sabudana", "rice", "sooji"],
    "corn flour": ["maida", "besan", "arrowroot", "potato starch"],
}
