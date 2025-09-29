# app.py
# Streamlit app — Google Maps Places + OpenAI (ChatGPT) classifier (step-by-step)

import os
import re
import time
import requests
import pandas as pd
import googlemaps
import openai
import streamlit as st
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()

# Read keys from env (set these in Streamlit Cloud secrets or your .env)
MAPS_KEY = os.getenv("GOOGLE_MAPS_KEY", "").strip()
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # change if not available

st.set_page_config(page_title="Restaurant Classifier (ChatGPT)", layout="wide")
st.title("🍽️ Restaurant Classifier — Streamlit + ChatGPT (Step-by-step)")

# Sidebar: keys override
with st.sidebar:
    st.header("API keys / settings")
    maps_key_input = st.text_input("Google Maps API Key", value=MAPS_KEY, type="password")
    openai_key_input = st.text_input("OpenAI (ChatGPT) API Key", value=OPENAI_KEY, type="password")
    model_input = st.text_input("OpenAI model (optional)", value=OPENAI_MODEL)
    if st.button("Use these keys"):
        MAPS_KEY = maps_key_input.strip()
        OPENAI_KEY = openai_key_input.strip()
        OPENAI_MODEL = model_input.strip()
        st.success("Keys updated (in-memory)")

st.write("Maps key loaded:", bool(MAPS_KEY), " — OpenAI key loaded:", bool(OPENAI_KEY))
st.write("OpenAI model:", OPENAI_MODEL)

# validate minimal config
if not MAPS_KEY:
    st.warning("Set GOOGLE_MAPS_KEY in sidebar or environment to enable Places API.")
if not OPENAI_KEY:
    st.warning("Set OPENAI_API_KEY in sidebar or environment to enable ChatGPT calls.")

# Helpers
def expand_short_url_once(url: str, timeout=4):
    try:
        r = requests.get(url, allow_redirects=True, timeout=timeout)
        return r.url
    except Exception:
        return url

def extract_coordinates(url: str):
    """Try to extract lat/lng from many Google Maps URL shapes; returns (lat, lng, elapsed_s)."""
    start = time.time()
    if not url:
        return None, None, round(time.time()-start, 3)
    u = url.strip()
    # expand short links quickly
    if "maps.app.goo.gl" in u or "goo.gl" in u:
        u = expand_short_url_once(u)
    # pattern @lat,lng
    m = re.search(r'@([-+]?\d+\.\d+),([-+]?\d+\.\d+)', u)
    if m:
        return float(m.group(1)), float(m.group(2)), round(time.time()-start,3)
    # pattern !3dLAT!4dLNG
    m = re.search(r'!3d([-+]?\d+\.\d+)!4d([-+]?\d+\.\d+)', u)
    if m:
        return float(m.group(1)), float(m.group(2)), round(time.time()-start,3)
    # fallback any lat,long pair
    m = re.search(r'([-+]?\d{1,3}\.\d+)[, ]+([-+]?\d{1,3}\.\d+)', u)
    if m:
        lat, lng = float(m.group(1)), float(m.group(2))
        if -90 <= lat <= 90 and -180 <= lng <= 180:
            return lat, lng, round(time.time()-start,3)
    return None, None, round(time.time()-start,3)

def fetch_restaurants_places(lat, lng, maps_key, radius=3000, max_pages=3):
    """Return DataFrame of restaurants (handles pagination)."""
    client = googlemaps.Client(key=maps_key)
    all_results = []
    places = client.places_nearby(location=(lat,lng), radius=radius, type="restaurant")
    all_results.extend(places.get("results", []))
    pages = 0
    while places.get("next_page_token") and pages < max_pages:
        pages += 1
        # API requires short wait before next_page_token becomes valid
        time.sleep(2)
        places = client.places_nearby(page_token=places["next_page_token"])
        all_results.extend(places.get("results", []))
    rows = []
    for r in all_results:
        rows.append({
            "name": r.get("name",""),
            "address": r.get("vicinity",""),
            "rating": r.get("rating",""),
            "types": ", ".join(r.get("types", [])),
            "place_id": r.get("place_id",""),
            "map_url": f"https://www.google.com/maps/place/?q=place_id:{r.get('place_id','')}"
        })
    return pd.DataFrame(rows)

# OpenAI classifier
CATEGORIES_AR = [
    "مطاعم هندية",
    "مطاعم شاورما",
    "مطاعم لبنانية",
    "مطاعم خليجية",
    "مطاعم أسماك",
    "مطاعم برجر",
    "أخرى"
]

def classify_with_chatgpt(name: str, address: str, types: str, timeout_s=30):
    """Call OpenAI ChatCompletion to classify a single restaurant.
       Returns (category_string, elapsed_seconds) or (error_string, None)."""
    if not OPENAI_KEY:
        return "❌ OpenAI API key missing", None
    openai.api_key = OPENAI_KEY
    system_msg = (
        "أنت مساعد لتصنيف أسماء المطاعم. صنّف المطعم إلى أحد التصنيفات التالية بدقة: "
        + ", ".join(CATEGORIES_AR)
        + ". أجب فقط بالكلمة العربية المطابقة بالضبط (مثلاً: مطاعم هندية) دون شرح."
    )
    user_msg = f"Name: {name}\nAddress: {address}\nTypes: {types}"
    messages = [
        {"role":"system", "content": system_msg},
        {"role":"user", "content": user_msg}
    ]
    try:
        t0 = time.time()
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=20,
            temperature=0.0,
            request_timeout=timeout_s
        )
        elapsed = time.time()-t0
        # parse reply
        text = ""
        if "choices" in resp and len(resp["choices"])>0:
            text = resp["choices"][0]["message"]["content"].strip()
        # normalize: if reply equals one of categories exactly -> good
        for cat in CATEGORIES_AR:
            if cat == text:
                return cat, round(elapsed,2)
        # try to map common english words to categories
        low = text.lower()
        if "indian" in low or "هندي" in low:
            return "مطاعم هندية", round(elapsed,2)
        if "shawarma" in low or "شاورما" in low:
            return "مطاعم شاورما", round(elapsed,2)
        if "lebanese" in low or "لبناني" in low:
            return "مطاعم لبنانية", round(elapsed,2)
        if "gulf" in low or "khaleeji" in low or "خليج" in low:
            return "مطاعم خليجية", round(elapsed,2)
        if "fish" in low or "seafood" in low or "سمك" in low:
            return "مطاعم أسماك", round(elapsed,2)
        if "burger" in low or "برجر" in low:
            return "مطاعم برجر", round(elapsed,2)
        # otherwise, fallback to "أخرى"
        return "أخرى", round(elapsed,2)
    except Exception as e:
        return f"❌ Error: {e}", None

# Streamlit session state init
if "coords" not in st.session_state: st.session_state["coords"] = None
if "restaurants" not in st.session_state: st.session_state["restaurants"] = None
if "classified" not in st.session_state: st.session_state["classified"] = []
if "index" not in st.session_state: st.session_state["index"] = 0
if "maps_key" not in st.session_state: st.session_state["maps_key"] = MAPS_KEY
if "openai_key" not in st.session_state: st.session_state["openai_key"] = OPENAI_KEY
if "model" not in st.session_state: st.session_state["model"] = OPENAI_MODEL

# UI
st.markdown("### 1) Paste Google Maps URL (short or long) and press **Start**")
url = st.text_input("Google Maps URL", placeholder="https://www.google.com/maps/place/...")
if st.button("▶️ Start — Extract Coordinates"):
    lat, lng, t = extract_coordinates(url)
    if lat is None:
        st.error(f"Could not extract coordinates (took {t}s). Paste full address-bar URL or long Google Maps link.")
    else:
        st.session_state["coords"] = (lat, lng)
        st.success(f"Coordinates: {lat}, {lng}  (extraction {t}s)")

st.markdown("### 2) Fetch nearby restaurants (Places API)")
if st.button("➡️ Continue — Fetch Restaurants"):
    if not st.session_state["coords"]:
        st.error("No coordinates — run Start first.")
    else:
        lat, lng = st.session_state["coords"]
        try:
            df = fetch_restaurants_places(lat, lng, st.session_state["maps_key"] or MAPS_KEY)
            if df.empty:
                st.warning("No restaurants found.")
            else:
                st.session_state["restaurants"] = df
                st.session_state["classified"] = []
                st.session_state["index"] = 0
                st.success(f"Found {len(df)} restaurants (showing first 50).")
                st.dataframe(df[["name","address","rating","types"]].head(50))
        except Exception as e:
            st.error(f"Places API error: {e}")

st.markdown("### 3) Classify restaurants via ChatGPT (one-by-one)")
col1, col2 = st.columns([1,3])
with col1:
    if st.button("➡️ Classify Next"):
        if st.session_state["restaurants"] is None:
            st.error("No restaurants loaded. Run Fetch step.")
        else:
            idx = st.session_state["index"]
            if idx >= len(st.session_state["restaurants"]):
                st.success("All restaurants classified.")
            else:
                row = st.session_state["restaurants"].iloc[idx]
                cat, t = classify_with_chatgpt(row["name"], row["address"], row["types"], timeout_s=45)
                st.session_state["classified"].append({
                    "name": row["name"],
                    "address": row["address"],
                    "category": cat,
                    "map_url": row.get("map_url",""),
                    "time_s": t
                })
                st.session_state["index"] += 1
                st.success(f"Classified {row['name']} -> {cat} (took {t}s)")
with col2:
    st.write("Progress:", st.session_state["index"], "/", len(st.session_state["restaurants"]) if st.session_state["restaurants"] is not None else 0)
    st.write("If you prefer classify-all at once, use the button below (may be slow).")

if st.button("Classify All (may take time)"):
    if st.session_state["restaurants"] is None:
        st.error("No restaurants loaded.")
    else:
        for i in range(st.session_state["index"], len(st.session_state["restaurants"])):
            row = st.session_state["restaurants"].iloc[i]
            cat, t = classify_with_chatgpt(row["name"], row["address"], row["types"], timeout_s=45)
            st.session_state["classified"].append({
                "name": row["name"], "address": row["address"], "category": cat, "map_url": row.get("map_url",""), "time_s": t
            })
            st.session_state["index"] += 1
            st.write(f"{i+1}/{len(st.session_state['restaurants'])}: {row['name']} -> {cat} ({t}s)")

# show classified table
if st.session_state["classified"]:
    st.subheader("Classified so far")
    st.dataframe(pd.DataFrame(st.session_state["classified"]))
