# build_indices.py
from utils import load_products, build_faiss_index

products = load_products("sample_data/products.csv")
texts = [row['title'] for _, row in products.iterrows()]

# OpenAI embeddings index
build_faiss_index(texts, model="openai", save_path="product_index_openai.faiss")

# Local embeddings index
build_faiss_index(texts, model="local", save_path="product_index_local.faiss")
