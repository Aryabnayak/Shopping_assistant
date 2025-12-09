# utils.py
import os
import numpy as np
import faiss
import pandas as pd
import sqlite3
import json

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_products(csv_path="sample_data/products.csv"):
    return pd.read_csv(csv_path)

# Embedding abstraction
def get_embeddings(texts, model="openai" , openai_client=None):
    """
    texts: list[str]
    model: "openai" or "local"
    If model == "openai", openai_client must be the openai python module already configured.
    Returns: np.ndarray (n, d)
    """
    if model == "openai":
        if openai_client is None:
            raise ValueError("openai_client required for openai embeddings")
        # Use text-embedding-3-small or similar
        resp = openai_client.Embedding.create(model="text-embedding-3-small", input=texts)
        embs = [r["embedding"] for r in resp["data"]]
        return np.array(embs).astype("float32")
    else:
        # sentence-transformers fallback
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        embs = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embs.astype("float32")

# FAISS helpers
def build_faiss_index(embeddings: np.ndarray, index_path="product_index.faiss"):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product with normalized vectors => cosine similarity if normalized
    # Normalize embeddings for cosine
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index

def load_faiss_index(index_path="product_index.faiss"):
    return faiss.read_index(index_path)

def create_metadata_db(products_df, db_path="products_meta.db"):
    # simple sqlite storing metadata (id -> product JSON)
    conn = sqlite3.connect(db_path)
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
