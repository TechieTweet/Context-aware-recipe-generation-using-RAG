import json
import os

# This works no matter where you run the script from
base = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(base, "flavordb.json")

with open(filepath, "r") as f:
    data = json.load(f)

print(f"Total ingredients: {len(data)}")
print(f"\nSample entry:")
print(json.dumps(data[0], indent=2))

categories = {}
for item in data:
    cat = item["category"]
    categories[cat] = categories.get(cat, 0) + 1

print(f"\nCategories found:")
for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")
