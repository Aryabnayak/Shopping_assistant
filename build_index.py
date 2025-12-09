# build_index.py
import os
import pickle
import argparse
import openai
import pandas as pd
from utils import load_products, get_embeddings, build_faiss_index, create_metadata_db

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="sample_data/products.csv")
parser.add_argument("--index", default="product_index.faiss")
parser.add_argument("--meta", default="products_meta.db")
parser.add_argument("--emb-method", choices=["openai","local"], default="openai")
args = parser.parse_args()

products = load_products(args.csv)
texts = (products["title"].fillna("") + ". " + products["description"].fillna("")).tolist()

if args.emb_method == "openai":
    if os.getenv("OPENAI_API_KEY") is None:
        raise RuntimeError("OPENAI_API_KEY must be set for openai embeddings or use --emb-method local")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    embs = get_embeddings(texts, model="openai", openai_client=openai)
else:
    embs = get_embeddings(texts, model="local", openai_client=None)

print("Embeddings shape:", embs.shape)
build_faiss_index(embs, index_path=args.index)
create_metadata_db(products, db_path=args.meta)
print("Index and metadata created.")
