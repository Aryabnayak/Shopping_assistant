# utils.py
import os
import numpy as np
import faiss
import pandas as pd
import sqlite3
import json
import openai
import numpy as np


def load_products(csv_path="sample_data/products.csv"):
    return pd.read_csv(csv_path)

def fetch_product_by_id(product_id, db_path="products_meta.db"):
    # Simple mock: return product from CSV (you can implement SQLite if needed)
    df = load_products()
    row = df[df['id'] == int(product_id)]
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        "id": str(row['id']),
        "title": row['title'],
        "price": float(row['price']),
        "category": row['category'],
        "image_url": row['image_url']
    }

def get_embeddings(texts, model="openai"):
    """
    Returns embeddings for list of texts.
    - OpenAI: 1536-d
    - Local (sentence-transformers): 384-d
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if model == "openai" and api_key:
        openai.api_key = api_key
        resp = openai.Embedding.create(
            model="text-embedding-3-small",
            input=texts
        )
        embs = [r["embedding"] for r in resp["data"]]

    elif model == "local":
        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            embs = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        except ImportError:
            # fallback random embeddings
            embs = [np.random.rand(384) for _ in texts]

    else:
        # fallback random embeddings
        embs = [np.random.rand(1536) for _ in texts]

    embs = np.array(embs, dtype=np.float32)
    embs = np.ascontiguousarray(embs)
    return embs

# FAISS helpers
import faiss

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def topk_products_from_index(index, query_emb, k=8):
    """
    Returns top-k IDs and similarity scores
    """
    # FAISS expects 2D array
    q = np.expand_dims(query_emb, axis=0).astype(np.float32)
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    return I[0], D[0]

# Build FAISS index
def build_faiss_index(texts, model="openai", save_path="product_index.faiss"):
    embs = get_embeddings(texts, model=model)
    faiss.normalize_L2(embs)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine similarity
    index.add(embs)
    faiss.write_index(index, save_path)
    print(f"FAISS index saved to {save_path}")
    return index

def create_metadata_db(products_df, db_path="products_meta.db"):
    # simple sqlite storing metadata (id -> product JSON)
    conn = sqlite3.connect(db_path)
    products_df.to_sql("products", conn, if_exists="replace", index=False)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS products (id TEXT PRIMARY KEY, json TEXT)""")
    for _, row in products_df.iterrows():
        pid = str(row["id"])
        data = {
            "id": pid,
            "title": row["title"],
            "category": row["category"],
            "description": row["description"],
            "price": float(row["price"]),
            "image_url": row["image_url"],
            "attributes": row["attributes"]
        }
        c.execute("INSERT OR REPLACE INTO products (id,json) VALUES (?,?)", (pid, json.dumps(data)))
    conn.commit()
    conn.close()
    return db_path



def fetch_product_by_id(pid, db_path="products_meta.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT json FROM products WHERE id=?", (str(pid),))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return json.loads(row[0])

def topk_products_from_index(index, query_emb, k=6):
    # query_emb shape (d,) or (1,d)
    import numpy as np
    q = np.copy(query_emb)
    if q.ndim == 1:
        q = q[np.newaxis, :]
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    # return indices and similarities
    return I[0].tolist(), D[0].tolist()
