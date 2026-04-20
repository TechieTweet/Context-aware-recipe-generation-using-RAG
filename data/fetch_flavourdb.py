import requests
import json
import time

BASE_URL = "https://cosylab.iiitd.edu.in/flavordb/entities_json?id={}"

ingredients = []

# FlavorDB has ~936 ingredients (IDs 1 to 1000, some missing)
for i in range(1, 1000):
    try:
        r = requests.get(BASE_URL.format(i), timeout=5)
        if r.status_code == 200:
            data = r.json()
            ingredients.append({
                "id": i,
                "name": data.get("entity_alias_readable", ""),
                "category": data.get("category", ""),
                "flavor_molecules": [m["common_name"] for m in data.get("molecules", [])]
            })
            print(f"Fetched {i}: {data.get('entity_alias_readable', '')}")
        time.sleep(0.2)  # be polite to the server
    except Exception as e:
        print(f"Skipped {i}: {e}")

with open("person3/data/flavordb.json", "w") as f:
    json.dump(ingredients, f, indent=2)

print(f"\nDone. Fetched {len(ingredients)} ingredients.")
