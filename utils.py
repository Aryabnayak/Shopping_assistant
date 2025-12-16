import os
import json
import re
import random
import base64
import numpy as np
import pandas as pd
import faiss
import openai
import requests
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer


# -------------------------------------------------
# Module 16: Silent Recovery Commerce™ (New)
# -------------------------------------------------
class SilentRecoveryService:
    """
    Implements Silent Recovery Commerce™.
    USP: "Problems are fixed before you notice them."
    Monitors: Delivery delays, Weather shifts, Stock issues.
    """

    @staticmethod
    def monitor_shipping_delays(orders):
        """
        Scans active orders. If a delay is simulated/detected on Standard shipping,
        automatically upgrade to Express/Hyper-Drone to meet the promise.
        """
        alerts = []
        for order in orders:
            # Only check active orders not already upgraded
            if "shipping_upgraded" not in order and order.get("shipping_method") == "Standard":
                # Simulate a logistics delay (e.g., 20% chance)
                if random.random() > 0.8:
                    order["shipping_method"] = "Hyper-Drone (Auto-Upgraded)"
                    order["shipping_upgraded"] = True
                    alerts.append(
                        f"🛡️ Recovery: Order #{order['order_id']} faced a delay. We auto-upgraded shipping to Hyper-Drone (Free) to ensure on-time arrival.")
        return alerts

    @staticmethod
    def monitor_weather_conflicts(cart, weather_context):
        """
        Checks if cart items are suitable for the current/forecasted weather.
        """
        alerts = []
        condition = weather_context.get("condition", "Sunny")

        for item in cart:
            # Reuse MaterialAnalyzer to check suitability
            analysis = MaterialAnalyzer.analyze(item.get("description", ""), condition)
            # If there are warnings, trigger a silent recovery suggestion
            if analysis["warnings"]:
                alerts.append(
                    f"🛡️ Adaptation: Weather shifted to {condition}. {item['title']} might be unsuitable. {analysis['warnings'][0]}")

        return alerts

    @staticmethod
    def monitor_stock_levels(cart):
        """
        Simulates low stock for items in cart and auto-reserves them.
        """
        alerts = []
        for item in cart:
            # Simulate low stock event (10% chance)
            if random.random() > 0.9 and not item.get("stock_reserved", False):
                item["stock_reserved"] = True
                alerts.append(
                    f"🛡️ Stock Watch: High demand detected for {item['title']}. We have silently reserved it for you for 15 mins.")
        return alerts


# -------------------------------------------------
# Module 15: Intent-Locked Pricing Service
# -------------------------------------------------
class PriceLockService:
    """
    Manages Intent-Locked Pricing™.
    1. Locks price on intent (Cart Add).
    2. Simulates market fluctuations.
    3. Calculates refunds if current price < locked price.
    """

    @staticmethod
    def get_market_price(original_price):
        """
        Simulates a live market price check.
        """
        # 30% chance that the market price has dropped significantly
        if random.random() > 0.7:
            # Drop price by 5% to 20%
            discount_factor = random.uniform(0.80, 0.95)
            return round(original_price * discount_factor, 2)

        return original_price

    @staticmethod
    def calculate_protection_refund(orders):
        """
        Scans past orders to see if price dropped within X days.
        Returns total refund amount.
        """
        total_refund = 0.0
        refund_details = []

        for order in orders:
            # Check if order is within 30 days
            if isinstance(order['date'], str):
                order_date = datetime.strptime(order['date'], "%Y-%m-%d").date()
            else:
                order_date = order['date']

            if (datetime.now().date() - order_date).days <= 30:
                for item in order['items']:
                    locked_price = item.get('locked_price', item['price'])
                    current_market = PriceLockService.get_market_price(locked_price)

                    if current_market < locked_price:
                        diff = locked_price - current_market
                        total_refund += diff
                        refund_details.append(f"{item['title']}: Dropped to ${current_market}")

        return round(total_refund, 2), refund_details


# -------------------------------------------------
# Module 12: Trend Forecasting Service
# -------------------------------------------------
class TrendService:
    @staticmethod
    def get_trends(location, season):
        trends_db = {
            "Summer": ["Linen", "Pastel", "Floral", "Oversized Tees", "Bucket Hats"],
            "Winter": ["Puffer Jackets", "Cashmere", "Turtlenecks", "Layering", "Boots"],
            "Spring": ["Denim", "Light Layers", "Trench Coats", "Sneakers"],
            "Fall": ["Leather", "Earth Tones", "Knits", "Scarves"]
        }

        loc_trends = []
        if location == "JP": loc_trends = ["Harajuku", "Minimalist"]
        if location == "US": loc_trends = ["Streetwear", "Athleisure"]
        if location == "EU": loc_trends = ["Chic", "Tailored"]

        base_trends = trends_db.get(season, ["Casual"])
        combined = list(set(base_trends + loc_trends))
        return combined


# -------------------------------------------------
# Module 13: Material & Fabric Analysis
# -------------------------------------------------
class MaterialAnalyzer:
    FABRIC_RULES = {
        "polyester": {"breathable": False, "warm": True, "weather": ["Cold", "Rainy", "Winter"]},
        "cotton": {"breathable": True, "warm": False, "weather": ["Sunny", "Summer", "Spring"]},
        "linen": {"breathable": True, "warm": False, "weather": ["Sunny", "Summer"]},
        "wool": {"breathable": True, "warm": True, "weather": ["Winter", "Cold", "Fall"]},
        "hemp": {"breathable": True, "warm": False, "weather": ["Sunny", "Summer"]}
    }

    @staticmethod
    def analyze(description, current_weather_condition):
        desc_lower = description.lower()
        warnings = []
        endorsements = []

        for fabric, props in MaterialAnalyzer.FABRIC_RULES.items():
            if fabric in desc_lower:
                if current_weather_condition in props["weather"]:
                    endorsements.append(f"✅ {fabric.capitalize()} is great for {current_weather_condition} weather.")
                else:
                    if current_weather_condition in ["Sunny", "Summer"] and props["warm"]:
                        warnings.append(f"⚠️ {fabric.capitalize()} might be too warm for current weather.")
                    elif current_weather_condition in ["Winter", "Cold"] and not props["warm"]:
                        warnings.append(f"⚠️ {fabric.capitalize()} might not be warm enough.")

        return {"warnings": warnings, "endorsements": endorsements}


# -------------------------------------------------
# Module 14: Cart Optimizer & Replenishment
# -------------------------------------------------
class CartOptimizer:
    @staticmethod
    def check_shipping_threshold(cart_total, threshold=50.0):
        if cart_total < threshold:
            return threshold - cart_total
        return 0


class ReplenishmentService:
    @staticmethod
    def predict_next_buy(purchase_history):
        suggestions = []
        today = datetime.now().date()

        for order in purchase_history:
            order_date = order.get("date")
            if not order_date: continue

            if isinstance(order_date, str):
                try:
                    order_date = datetime.strptime(order_date, "%Y-%m-%d").date()
                except:
                    continue

            days_diff = (today - order_date).days

            for item in order.get("items", []):
                if any(x in item['title'].lower() for x in ['cream', 'lotion', 'shampoo', 'serum']):
                    if 25 <= days_diff <= 35:
                        suggestions.append(item)
        return suggestions


# -------------------------------------------------
# Module 11: Image Processing Utils
# -------------------------------------------------
def encode_image(file_obj):
    if file_obj is None:
        return None
    try:
        file_obj.seek(0)
        return base64.b64encode(file_obj.read()).decode('utf-8')
    except Exception as e:
        print(f"Image Encoding Error: {e}")
        return None


# -------------------------------------------------
# Module 10: External Review Aggregator (Google API)
# -------------------------------------------------
class GoogleReviewService:
    API_KEY = os.getenv("GOOGLE_API_KEY")
    CSE_ID = os.getenv("GOOGLE_CSE_ID")

    @staticmethod
    def fetch_rating(product_title):
        if not GoogleReviewService.API_KEY or not GoogleReviewService.CSE_ID:
            return GoogleReviewService._simulate_rating(product_title)

        try:
            query = f"{product_title} product reviews rating"
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'q': query,
                'key': GoogleReviewService.API_KEY,
                'cx': GoogleReviewService.CSE_ID,
                'num': 3
            }

            resp = requests.get(url, params=params, timeout=2)
            if resp.status_code != 200:
                return GoogleReviewService._simulate_rating(product_title)

            data = resp.json()
            total_score = 0
            count = 0

            for item in data.get("items", []):
                snippet = item.get("snippet", "") + item.get("title", "")
                match = re.search(r"(\d(\.\d)?)\s*(?:/|out of)\s*5", snippet)
                if match:
                    score = float(match.group(1))
                    total_score += score
                    count += 1

            if count > 0:
                avg_rating = round(total_score / count, 1)
                return {"rating": avg_rating, "source": "Google Verified", "count": count}
            else:
                return GoogleReviewService._simulate_rating(product_title)

        except Exception as e:
            return GoogleReviewService._simulate_rating(product_title)

    @staticmethod
    def _simulate_rating(product_title):
        seed = sum(ord(c) for c in product_title)
        random.seed(seed)
        rating = round(random.uniform(3.8, 5.0), 1)
        count = random.randint(10, 500)
        return {"rating": rating, "source": "Reviews", "count": count}


# -------------------------------------------------
# Module 9: Weather & Location Service
# -------------------------------------------------
class WeatherService:
    @staticmethod
    def get_context():
        try:
            location_response = requests.get("https://ipinfo.io/json", timeout=3)
            if location_response.status_code == 200:
                location_data = location_response.json()
                loc_str = location_data.get("loc", "0,0")
                if "," in loc_str:
                    lat, lon = loc_str.split(",")
                else:
                    lat, lon = "0", "0"

                city = location_data.get("city", "Unknown City")
                country = location_data.get("country", "US")
            else:
                raise Exception("Location API unavailable")

            weather_response = requests.get(
                f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&timezone=auto",
                timeout=3
            )

            if weather_response.status_code == 200:
                weather_data = weather_response.json()
                current = weather_data.get("current_weather", {})
                temp = current.get("temperature", 20)
                weathercode = current.get("weathercode", 0)

                season = WeatherService._infer_season(temp)
                condition = WeatherService._infer_condition_text(weathercode)

                return {
                    "city": city,
                    "country": country,
                    "temp": temp,
                    "season": season,
                    "condition": condition,
                    "success": True
                }
            else:
                raise Exception("Weather API unavailable")

        except Exception as e:
            return WeatherService._get_fallback_context()

    @staticmethod
    def _infer_season(temp):
        if temp > 25:
            return "Summer"
        elif temp > 18:
            return "Spring"
        elif temp > 10:
            return "Fall"
        else:
            return "Winter"

    @staticmethod
    def _infer_condition_text(code):
        if code > 60:
            return "Rainy"
        elif code > 40:
            return "Cloudy"
        return "Sunny"

    @staticmethod
    def _get_fallback_context():
        return {
            "city": "Local Area",
            "country": "US",
            "temp": 20,
            "season": "Spring",
            "condition": "Sunny",
            "success": False
        }


# -------------------------------------------------
# Module 8: Shipping & Return Policy Logic
# -------------------------------------------------
class PolicyManager:
    SHIPPING_OPTIONS = {
        "Standard": {"cost": 0, "days": "5-7 Business Days", "threshold": 50},
        "Express": {"cost": 15.00, "days": "2-3 Business Days", "threshold": 100},
        "Hyper-Drone": {"cost": 25.00, "days": "1 Business Day", "threshold": 200}
    }

    @staticmethod
    def get_shipping_cost(method):
        return PolicyManager.SHIPPING_OPTIONS.get(method, {}).get("cost", 0)

    @staticmethod
    def get_free_shipping_threshold(method):
        return PolicyManager.SHIPPING_OPTIONS.get(method, {}).get("threshold", 999)

    @staticmethod
    def check_return_eligibility(purchase_date):
        if not purchase_date:
            return False, "Date not found."

        if isinstance(purchase_date, str):
            try:
                purchase_date = datetime.strptime(purchase_date, "%Y-%m-%d").date()
            except:
                pass

        today = datetime.now().date()
        if isinstance(purchase_date, datetime):
            purchase_date = purchase_date.date()

        delta = today - purchase_date

        if delta.days <= 15:
            return True, f"Eligible ({delta.days} days since purchase)"
        else:
            return False, f"Ineligible ({delta.days} days since purchase. Policy: 15 days)"


# -------------------------------------------------
# Module 6: Sizes as per Location Enabled
# -------------------------------------------------
class SizeConverter:
    SIZE_MAP = {
        "US": {"XS": "0-2", "S": "4-6", "M": "8-10", "L": "12-14", "XL": "16+"},
        "EU": {"XS": "32-34", "S": "36-38", "M": "40-42", "L": "44-46", "XL": "48+"},
        "UK": {"XS": "4-6", "S": "8-10", "M": "12-14", "L": "16-18", "XL": "20+"},
        "JP": {"XS": "5-7", "S": "9", "M": "11", "L": "13", "XL": "15"},
    }

    @staticmethod
    def convert(size_key, region="US"):
        region_map = SizeConverter.SIZE_MAP.get(region, SizeConverter.SIZE_MAP["US"])
        local_size = region_map.get(size_key, size_key)
        return f"{region} {local_size} ({size_key})"


# -------------------------------------------------
# Module 7: Customer Rewards Logic
# -------------------------------------------------
class RewardSystem:
    @staticmethod
    def calculate_points(action_type, amount=0):
        if action_type == "purchase":
            return int(amount)
        rewards = {
            "review": 50,
            "share": 25,
            "ar_try_on": 10,
            "eco_choice": 15
        }
        return rewards.get(action_type, 0)


# -------------------------------------------------
# Load products from CSV
# -------------------------------------------------
def load_products(csv_path="sample_data/products.csv"):
    if not os.path.exists(csv_path):
        data = {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "title": ["Eco-Hemp Jacket", "Urban Glide Sneakers", "Midnight Gala Dress", "Smart Wool Blazer",
                      "Boho Summer Tunic", "Basic Cotton Tee", "Silk Scarf", "Running Socks"],
            "category": ["Men's Fashion", "Footwear", "Women's Fashion", "Men's Fashion", "Women's Fashion", "Unisex",
                         "Accessories", "Accessories"],
            "description": ["Sustainable hemp fiber jacket.", "High-tech running shoes.", "Elegant evening wear.",
                            "Professional office attire (Wool).", "Lightweight summer vibe (Linen).",
                            "100% Organic Cotton.", "Pure Silk.", "Moisture wicking synthetic."],
            "price": [120.00, 85.00, 250.00, 180.00, 45.00, 25.00, 40.00, 12.00],
            "image_url": ["", "", "", "", "", "", "", ""],
            "attributes": ["Sustainable", "Sporty", "Formal", "Business", "Casual", "Basic", "Luxury", "Sport"]
        }
        return pd.DataFrame(data)
    return pd.read_csv(csv_path)


def fetch_product_by_id(product_id, csv_path="sample_data/products.csv"):
    df = load_products(csv_path)
    try:
        row = df[df["id"].astype(str) == str(product_id)]
    except KeyError:
        return None

    if row.empty: return None

    row = row.iloc[0]
    return {
        "id": str(row["id"]),
        "title": row["title"],
        "category": row["category"],
        "description": row["description"],
        "price": float(row["price"]),
        "image_url": row.get("image_url", ""),
        "attributes": row.get("attributes", "")
    }


# -------------------------------------------------
# Embeddings & FAISS
# -------------------------------------------------
def get_embeddings(texts, model="openai"):
    api_key = os.getenv("OPENAI_API_KEY")
    if model == "openai" and api_key:
        openai.api_key = api_key
        try:
            resp = openai.Embedding.create(model="text-embedding-3-small", input=texts)
            embs = [r["embedding"] for r in resp["data"]]
        except Exception as e:
            print(f"OpenAI Error: {e}, falling back.")
            embs = [np.random.rand(1536) for _ in texts]
    elif model == "local":
        try:
            embedder = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device="cpu"
            )
            embs = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        except ImportError:
            embs = [np.random.rand(384) for _ in texts]
    else:
        embs = [np.random.rand(1536) for _ in texts]

    embs = np.array(embs, dtype=np.float32)
    return np.ascontiguousarray(embs)


def load_faiss_index(index_path):
    try:
        return faiss.read_index(index_path)
    except:
        return None


def build_faiss_index(texts, model="openai", save_path="product_index.faiss"):
    embs = get_embeddings(texts, model=model)
    faiss.normalize_L2(embs)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    faiss.write_index(index, save_path)
    return index


def topk_products_from_index(index, query_emb, k=6):
    q = np.array(query_emb, dtype=np.float32)
    if q.ndim == 1: q = q[np.newaxis, :]
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    return I[0].tolist(), D[0].tolist()