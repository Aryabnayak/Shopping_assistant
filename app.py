# app.py
import streamlit as st
from streamlit import rerun   # ✅ NEW replacement for experimental_rerun
from agent import ShoppingAgent
from utils import load_products, fetch_product_by_id
import os
import json

st.set_page_config(layout="wide", page_title="AI Personal Shopper — Hyper-Personalized")

# Initialize (once)
if "agent" not in st.session_state:
    emb_method = "openai" if os.getenv("OPENAI_API_KEY") else "local"
    st.session_state.agent = ShoppingAgent(
        index_path="product_index.faiss",
        meta_db="products_meta.db",
        emb_method=emb_method
    )

if "history" not in st.session_state:
    st.session_state.history = []  # list of (role,text)

if "cart" not in st.session_state:
    st.session_state.cart = []

if "orders" not in st.session_state:
    st.session_state.orders = []

products_df = load_products("sample_data/products.csv")

# Sidebar: user profile/preferences
with st.sidebar:
    st.title("Preferences")
    occasion = st.selectbox("Occasion", ["wedding", "party", "office", "casual", "travel"], index=0)
    temp = st.selectbox("Weather", ["warm", "mild", "chilly", "cold"], index=2)
    size = st.selectbox("Size", ["XS", "S", "M", "L", "XL"], index=2)
    budget_min, budget_max = st.slider("Budget range (USD)", 20, 1000, (50, 300))

    st.markdown("---")
    st.subheader("Cart")

    if st.session_state.cart:
        for i, item in enumerate(st.session_state.cart):
            st.markdown(f"**{item['title']}** — ${item['price']}")
            if st.button(f"Remove {i}", key=f"remove_{i}"):
                st.session_state.cart.pop(i)
                rerun()   # ✅ FIXED

        st.markdown(f"**Total:** ${sum([c['price'] for c in st.session_state.cart]):.2f}")

        if st.button("Checkout"):
            st.session_state.show_checkout = True
    else:
        st.write("Cart is empty")

    st.markdown("---")
    st.write("Order history")
    for o in st.session_state.orders[::-1]:
        st.write(f"Order #{o['order_id']} — ${o['total']:.2f}")

# Main UI
col1, col2 = st.columns([2, 3])

with col1:
    st.header("Chat with your Personal Shopper")

    # Show conversation
    for role, text in st.session_state.history:
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Assistant:** {text}")

    user_input = st.text_input(
        "Tell me what you need (e.g., 'An outfit for a chilly, semi-formal outdoor wedding')",
        key="input"
    )

    if st.button("Send"):
        st.session_state.history.append(("user", user_input))

        # Build retrieval query enriched with preferences
        query = (
            f"{user_input}. Occasion: {occasion}. Weather: {temp}. "
            f"Budget between {budget_min} and {budget_max} USD. Size: {size}."
        )

        agent = st.session_state.agent
        retrieved, sims = agent.retrieve(query, k=12)
        st.session_state.retrieved = retrieved

        parsed = agent.generate_lookbook(
            user_request=query,
            retrieved_products=retrieved,
            chat_history=[m for _, m in st.session_state.history]
        )

        st.session_state.history.append(("assistant", "I created a curated lookbook. Scroll right to see it."))
        st.session_state.last_lookbook = parsed

        rerun()   # ✅ FIXED

with col2:
    st.header("Curated Lookbook")

    lookbook = st.session_state.get("last_lookbook", {})

    if lookbook:
        items = lookbook.get("lookbook", [])

        if not items:
            st.write("No items in lookbook. Try a different request.")
        else:
            cols = st.columns(2)

            for i, it in enumerate(items):
                pid = it.get("product_id") or it.get("id")
                product = fetch_product_by_id(pid)

                if not product:
                    continue

                col = cols[i % len(cols)]
                with col:
                    st.image(product["image_url"], width=250)
                    st.markdown(f"**{product['title']}**")
                    st.write(f"${product['price']:.2f} • {product['category']}")
                    st.write(it.get("reason", ""))

                    if st.button("Add to cart", key=f"add_{pid}"):
                        st.session_state.cart.append(product)
                        st.success("Added to cart")

            # Styling notes + complementary items
            st.markdown("---")
            st.subheader("Styling notes")
            st.write(lookbook.get("styling_notes", ""))

            st.subheader("Complementary items")
            for c in lookbook.get("complementary_items", []):
                st.write(f"- {c.get('title')} — {c.get('reason', '')}")

            st.markdown("---")
            st.subheader("Checkout instructions")
            st.write(lookbook.get("checkout_instructions", ""))

    else:
        st.write("No lookbook generated yet. Ask for an outfit on the left.")

# Checkout Section
if st.session_state.get("show_checkout"):
    st.sidebar.markdown("## Checkout")
    name = st.sidebar.text_input("Name")
    email = st.sidebar.text_input("Email")
    addr = st.sidebar.text_area("Shipping Address")

    if st.sidebar.button("Place Order"):
        order_id = len(st.session_state.orders) + 1
        total = sum([c['price'] for c in st.session_state.cart])

        st.session_state.orders.append({
            "order_id": order_id,
            "items": st.session_state.cart.copy(),
            "total": total,
            "name": name,
            "email": email,
            "address": addr
        })

        st.session_state.cart = []
        st.session_state.show_checkout = False
        st.sidebar.success(f"Order #{order_id} placed. Total ${total:.2f}")

# Returns + Post-purchase Recs
st.sidebar.markdown("---")
st.sidebar.subheader("Manage Returns / Post-purchase Recs")

if st.sidebar.button("Recommend for last order"):
    if st.session_state.orders:
        last = st.session_state.orders[-1]
        recs = st.session_state.agent.post_purchase_recommendations(last["items"], top_n=4)
        st.sidebar.write("Recommended complementary items:")
        for r in recs:
            st.sidebar.write(f"- {r['title']} — ${r['price']}")
    else:
        st.sidebar.write("No orders yet.")

if st.sidebar.button("Initiate return (last order)"):
    if st.session_state.orders:
        last = st.session_state.orders.pop()
        st.sidebar.success(f"Return processed for Order #{last['order_id']}. Refund ${last['total']:.2f}")
    else:
        st.sidebar.write("No orders to return.")
