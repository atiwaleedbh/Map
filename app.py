# app.py
import os
import re
import time
import requests
import pandas as pd
import googlemaps
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

# Load .env if present (for local run)
load_dotenv()

# Read keys from environment (recommended). When deploying to Streamlit Cloud, set these in the app secrets.
MAPS_KEY = os.getenv("GOOGLE_MAPS_KEY", "")
GEMINI_KEY = os.getenv("GEMINI_KEY", "")

st.set_page_config(page_title="Restaurant Classifier", layout="wide")
st.title("🍴 Restaurant Classifier — Step by Step (Streamlit)")

st.markdown("**Flow:** Paste Google Maps URL → Start (get coords) → Fetch restaurants → Classify next restaurant (one click per restaurant).")

# Sidebar: allow overriding keys or show status
with st.sidebar:
    st.header("API keys")
    maps_key_input = st.text_input("Google Maps API Key", value=MAPS_KEY, type="password")
    gemini_key_input = st.text_input("Gemini (AI) API Key", value=GEMINI_KEY, type="password")
    use_keys_btn = st.button("Use these keys")

if use_keys_btn:
    MAPS_KEY = maps_key_input.strip()
    GEMINI_KEY = gemini_key_input.strip()
    st.success("Keys updated (in-memory).")

st.write("Maps key loaded:", bool(MAPS_KEY), " — Gemini key loaded:", bool(GEMINI_KEY))

# Helpers
def expand_short_url(url):
    if "maps.app.goo.gl" in url or "goo.gl" in url:
        try:
            r = requests.get(url, allow_redirects=True, timeout=5)
            return r.url
        except Exception:
            return url
    return url

def extract_coordinates(url: str):
    """Return (lat, lng, elapsed_seconds) or (None, None, elapsed)."""
    start = time.time()
    if not url:
        return None, None, round(time.time()-start,2)
    final = url.strip()
    # expand short links first (fast attempt)
    if "maps.app.goo.gl" in final or "goo.gl" in final:
        final = expand_short_url(final)
    # common patterns
    m = re.search(r'@([-+]?\d+\.\d+),([-+]?\d+\.\d+)', final)
    if m:
        return float(m.group(1)), float(m.group(2)), round(time.time()-start,2)
    m = re.search(r'!3d([-+]?\d+\.\d+)!4d([-+]?\d+\.\d+)', final)
    if m:
        return float(m.group(1)), float(m.group(2)), round(time.time()-start,2)
    # fallback: any lat,long pair
    m = re.search(r'([-+]?\d{1,3}\.\d+)[, ]+([-+]?\d{1,3}\.\d+)', final)
    if m:
        lat, lng = float(m.group(1)), float(m.group(2))
        if -90 <= lat <= 90 and -180 <= lng <= 180:
            return lat, lng, round(time.time()-start,2)
    return None, None, round(time.time()-start,2)

def fetch_restaurants(lat, lng, maps_key, radius=3000):
    client = googlemaps.Client(key=maps_key)
    all_results = []
    places = client.places_nearby(location=(lat,lng), radius=radius, type="restaurant")
    all_results.extend(places.get("results", []))
    # pagination
    while places.get("next_page_token"):
        time.sleep(2)
        token = places["next_page_token"]
        places = client.places_nearby(page_token=token)
        all_results.extend(places.get("results", []))
    # normalize
    rows = []
    for r in all_results:
        rows.append({
            "name": r.get("name",""),
            "address": r.get("vicinity",""),
            "rating": r.get("rating", ""),
            "types": ", ".join(r.get("types", [])),
            "place_id": r.get("place_id",""),
            "map_url": f"https://www.google.com/maps/place/?q=place_id:{r.get('place_id','')}"
        })
    return pd.DataFrame(rows)

# Model helpers
def pick_model():
    genai.configure(api_key=GEMINI_KEY)
    models = list(genai.list_models())
    for m in models:
        sm = getattr(m, "supported_generation_methods", None) or getattr(m, "supportedGenerationMethods", None) or []
        if any("generate" in str(x).lower() for x in sm):
            return m.name
    # fallback
    return None

def classify_one(name, address, types, model_name, timeout_s=60):
    prompt = f"""Classify this restaurant into one of: Indian, Shawarma, Lebanese, Khaleeji, Seafood, Burger, Other.
Name: {name}
Address: {address}
Types: {types}
Reply only the category name."""
    try:
        model = genai.GenerativeModel(model_name)
        start = time.time()
        resp = model.generate_content(prompt, request_options={"timeout": timeout_s})
        elapsed = time.time() - start
        cat = resp.text.strip()
        return cat, round(elapsed,2)
    except Exception as e:
        return f"ERROR: {e}", None

# Session state
if "coords" not in st.session_state: st.session_state.coords = None
if "restaurants" not in st.session_state: st.session_state.restaurants = None
if "classified" not in st.session_state: st.session_state.classified = []
if "index" not in st.session_state: st.session_state.index = 0
if "model_name" not in st.session_state: st.session_state.model_name = None

# UI controls
col1, col2 = st.columns([2,3])
with col1:
    url = st.text_input("Google Maps URL (short or long)")
    if st.button("▶️ Start — Extract Coordinates"):
        lat, lng, elapsed = extract_coordinates(url)
        if lat is None:
            st.error(f"Could not extract coordinates (took {elapsed}s). Paste full address-bar URL.")
        else:
            st.session_state.coords = (lat, lng)
            st.success(f"Coordinates: {lat}, {lng}  — extraction {elapsed}s")
            # initialize model
            try:
                st.session_state.model_name = pick_model()
                if st.session_state.model_name:
                    st.info(f"Using Gemini model: {st.session_state.model_name}")
                else:
                    st.warning("No suitable model auto-detected. Ensure GEMINI_KEY is valid and account has access.")
            except Exception as e:
                st.error(f"Gemini init error: {e}")

    if st.session_state.coords:
        if st.button("➡️ Continue — Fetch Restaurants"):
            lat, lng = st.session_state.coords
            try:
                df = fetch_restaurants(lat, lng, maps_key_input or MAPS_KEY)
                st.session_state.restaurants = df
                st.session_state.classified = []
                st.session_state.index = 0
                st.success(f"Found {len(df)} restaurants (showing first 10).")
            except Exception as e:
                st.error(f"Places API error: {e}")

with col2:
    st.subheader("Status / Controls")
    st.write("Coordinates:", st.session_state.coords)
    st.write("Model:", st.session_state.model_name)
    st.write("To classify: press 'Classify Next' to classify one restaurant at a time.")

# Display restaurants table
if st.session_state.restaurants is not None:
    st.subheader("Restaurants")
    df_show = st.session_state.restaurants.copy()
    # show a clickable link (map_url) if present
    if "map_url" in df_show.columns:
        df_show["map_link"] = df_show["map_url"]
    st.dataframe(df_show[["name","address","rating","types"]].head(50))

# Classify next
if st.button("➡️ Classify Next"):
    if st.session_state.restaurants is None:
        st.error("No restaurants loaded. Run Fetch step first.")
    else:
        idx = st.session_state.index
        if idx >= len(st.session_state.restaurants):
            st.success("All restaurants classified.")
        else:
            row = st.session_state.restaurants.iloc[idx]
            model_name = st.session_state.model_name
            if not model_name:
                st.error("No Gemini model selected/available. Check GEMINI_KEY.")
            else:
                cat, t = classify_one(row["name"], row["address"], row["types"], model_name, timeout_s=60)
                st.session_state.classified.append({
                    "name": row["name"], "address": row["address"], "category": cat, "map_url": row.get("map_url",""), "time_s": t
                })
                st.session_state.index += 1
                st.success(f"Classified {row['name']} -> {cat} (took {t}s)")
                st.dataframe(pd.DataFrame(st.session_state.classified))

# Option: classify all at once (if desired)
if st.button("Classify All (careful: may take time)"):
    if st.session_state.restaurants is None:
        st.error("No restaurants loaded.")
    else:
        model_name = st.session_state.model_name
        for i in range(st.session_state.index, len(st.session_state.restaurants)):
            row = st.session_state.restaurants.iloc[i]
            cat, t = classify_one(row["name"], row["address"], row["types"], model_name, timeout_s=45)
            st.session_state.classified.append({"name": row["name"], "address": row["address"], "category": cat, "map_url": row.get("map_url",""), "time_s": t})
            st.session_state.index += 1
            st.write(f"{i+1}/{len(st.session_state.restaurants)}: {row['name']} -> {cat} ({t}s)")
        st.success("Classification finished.")
        st.dataframe(pd.DataFrame(st.session_state.classified))
