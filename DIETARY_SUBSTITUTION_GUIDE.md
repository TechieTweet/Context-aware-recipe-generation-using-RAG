# Dietary Substitution Features - Usage Guide

## Overview
The substitution module now supports filtering ingredient substitutes by dietary restrictions including vegan, vegetarian, and gluten-free alternatives.

## Basic Usage

### 1. Standard Substitutes (No Filtering)
```python
from substitution import get_substitutes

# Get top 5 substitutes for butter
substitutes = get_substitutes("butter", top_k=5)
```

### 2. Vegan Alternatives
```python
# Get only vegan substitutes for butter
vegan_subs = get_substitutes("butter", top_k=5, dietary_restrictions=['vegan'])
```

**Example Output:**
- coconut oil (highest flavor match)
- sesame oil
- groundnut oil
- sunflower oil

### 3. Gluten-Free Alternatives
```python
# Get only gluten-free substitutes for wheat
gluten_free_subs = get_substitutes("wheat", top_k=5, dietary_restrictions=['gluten_free'])
```

**Example Output:**
- rice
- ragi (finger millet)
- bajra (pearl millet)
- besan (chickpea flour)
- poha (flattened rice)

### 4. Multiple Dietary Restrictions (AND logic)
```python
# Get vegan AND gluten-free substitutes
dual_restrict = get_substitutes("butter", top_k=5, 
                               dietary_restrictions=['vegan', 'gluten_free'])
```

This will return only ingredients that are BOTH vegan AND gluten-free.

## Supported Dietary Restrictions
- `'vegan'` - Plant-based, excludes dairy, eggs, honey, meat
- `'vegetarian'` - Excludes meat, includes dairy and eggs (optional)
- `'gluten_free'` - Excludes wheat, barley, sooji, and other gluten-containing items

## Implementation Details

### New Files
1. **`data/dietary_properties.py`** - Contains:
   - `DIETARY_PROPERTIES` dict mapping ingredients to dietary attributes
   - `filter_by_dietary_restrictions()` function for filtering candidates
   - `get_dietary_status()` function to check individual ingredient properties

### Modified Files
1. **`substitution/substitution_model.py`** - Updated to:
   - Import dietary properties
   - Accept `dietary_restrictions` parameter in `get_substitutes()`
   - Filter candidates before ranking
   - Enhanced test section with dietary examples

2. **`substitution/substitution.py`** - Updated to:
   - Accept and pass through `dietary_restrictions` parameter
   - Updated docstring with usage examples

## Key Features
✅ **Backward Compatible** - Existing code works without dietary restrictions
✅ **Flexible** - Combine multiple dietary restrictions with AND logic
✅ **Non-Intrusive** - No changes to existing scoring algorithm
✅ **Comprehensive** - Covers all Indian and common ingredients
✅ **Accurate** - Properly classifies dairy, grains, oils, and other categories

## Example: Complete Integration

```python
from substitution import get_substitutes

def suggest_healthy_substitute(ingredient, dietary_needs=[]):
    """
    Suggest a healthy substitute with optional dietary restrictions.
    
    Args:
        ingredient: The ingredient to replace
        dietary_needs: List of dietary restrictions (e.g., ['vegan', 'gluten_free'])
    
    Returns:
        The best substitute matching the dietary needs
    """
    substitutes = get_substitutes(
        ingredient, 
        top_k=3, 
        dietary_restrictions=dietary_needs if dietary_needs else None
    )
    
    if substitutes:
        best = substitutes[0]
        print(f"Substitute for '{ingredient}': {best['ingredient']}")
        print(f"Flavor similarity: {best['flavor_similarity']}")
        print(f"Score: {best['final_score']}")
        return best['ingredient']
    else:
        print(f"No {' + '.join(dietary_needs)} substitutes found for '{ingredient}'")
        return None

# Usage examples
suggest_healthy_substitute("butter")  # Regular substitute
suggest_healthy_substitute("butter", ['vegan'])  # Vegan only
suggest_healthy_substitute("wheat", ['gluten_free'])  # Gluten-free only
suggest_healthy_substitute("ghee", ['vegan', 'gluten_free'])  # Both restrictions
```

## Testing
Run the included test file to verify the dietary filtering:
```bash
python test_dietary_filter.py
```

This will verify:
- Individual ingredient classification
- Filtering functions work correctly
- Combined dietary restrictions work as expected
