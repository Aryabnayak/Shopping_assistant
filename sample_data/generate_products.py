# sample_data/generate_products.py
import csv
import os
from random import choice, uniform, randint
from urllib.parse import quote_plus

OUT = "sample_data"
os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, "products.csv")

categories = {
    "Women's Dresses": ["maxi dress", "cocktail dress", "wrap dress", "midi dress"],
    "Men's Suits": ["two-piece suit", "blazer", "tuxedo", "morning coat"],
    "Outerwear": ["wool coat", "pea coat", "trench coat", "puffer jacket"],
    "Accessories": ["clutch", "leather belt", "silk scarf", "statement necklace"],
    "Shoes": ["heels", "oxfords", "loafers", "boots"]
}

colors = ["navy", "black", "ivory", "emerald", "burgundy", "charcoal", "tan", "blush"]
materials = ["wool", "silk", "cotton", "linen", "polyester blend", "suede", "leather"]

def unsplash_image_url(query, w=800, h=800):
    # unsplash source example (no API key needed for the random photo endpoints)
    q = quote_plus(query)
    return f"https://source.unsplash.com/{w}x{h}/?{q}"

rows = []
id_counter = 1
for cat, templates in categories.items():
    for i in range(20):  # 20 items per category => ~100 products
        template = choice(templates)
        color = choice(colors)
        mat = choice(materials)
        title = f"{color.title()} {template.title()} ({mat})"
        desc = (f"A {color} {template} made from {mat}. Versatile, elegant, "
                "and suitable for semi-formal events. Lightweight, breathable, and tailored fit.")
        price = round(uniform(39.99, 499.99), 2)
        img_q = f"{cat} {template} {color}"
        img_url = unsplash_image_url(img_q)
        attrs = {"color": color, "material": mat, "occasion": "semi-formal" if "dress" in template or "suit" in template else "casual"}
        rows.append({
            "id": str(id_counter),
            "title": title,
            "category": cat,
            "description": desc,
            "price": price,
            "image_url": img_url,
            "attributes": str(attrs)
        })
        id_counter += 1

with open(F, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=["id","title","category","description","price","image_url","attributes"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"Wrote {len(rows)} products to {F}")
