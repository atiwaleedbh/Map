# app.py — Streamlit + Google Maps + OpenAI ChatGPT classifier (updated)

import os, re, time, requests, pandas as pd, googlemaps, streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------
# 1️⃣ Load .env if present & secrets
# -------------------------
load_dotenv()
MAPS_KEY = os.getenv("GOOGLE_MAPS_KEY", "").strip()
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_KEY)

st.set_page_config(page_title="Restaurant Classifier", layout="wide")
st.title("🍽️ Restaurant Classifier — Streamlit + ChatGPT")

# Sidebar keys override
with st.sidebar:
    st.header("API Keys / Settings")
    maps_key_input = st.text_input("Google Maps API Key", value=MAPS_KEY, type="password")
    openai_key_input = st.text_input("OpenAI API Key", value=OPENAI_KEY, type="password")
    model_input = st.text_input("OpenAI Model (optional)", value=OPENAI_MODEL)
    if st.button("Update Keys"):
        MAPS_KEY = maps_key_input.strip()
        OPENAI_KEY = openai_key_input.strip()
        OPENAI_MODEL = model_input.strip()
        client.api_key = OPENAI_KEY
        st.success("Keys updated (in-memory)")

# -------------------------
# 2️⃣ Helpers
# -------------------------
def expand_short_url(url, timeout=4):
    try:
        r = requests.get(url, allow_redirects=True, timeout=timeout)
        return r.url
    except:
        return url

def extract_coordinates(url: str):
    start = time.time()
    if not url:
        return None, None, 0
    u = url.strip()
    if "maps.app.goo.gl" in u or "goo.gl" in u:
        u = expand_short_url(u)
    m = re.search(r'@([-+]?\d+\.\d+),([-+]?\d+\.\d+)', u)
    if m:
        return float(m.group(1)), float(m.group(2)), round(time.time()-start,3)
    m = re.search(r'!3d([-+]?\d+\.\d+)!4d([-+]?\d+\.\d+)', u)
    if m:
        return float(m.group(1)), float(m.group(2)), round(time.time()-start,3)
    return None, None, round(time.time()-start,3)

def fetch_restaurants(lat, lng, maps_key, radius=3000, max_pages=3):
    client_gm = googlemaps.Client(key=maps_key)
    all_results = []
    places = client_gm.places_nearby(location=(lat,lng), radius=radius, type="restaurant")
    all_results.extend(places.get("results",[]))
    pages = 0
    while places.get("next_page_token") and pages<max_pages:
        pages += 1
        time.sleep(2)
        places = client_gm.places_nearby(page_token=places["next_page_token"])
        all_results.extend(places.get("results",[]))
    rows = []
    for r in all_results:
        rows.append({
            "name": r.get("name",""),
            "address": r.get("vicinity",""),
            "rating": r.get("rating",""),
            "types": ", ".join(r.get("types",[])),
            "place_id": r.get("place_id",""),
            "map_url": f"https://www.google.com/maps/place/?q=place_id:{r.get('place_id','')}"
        })
    return pd.DataFrame(rows)

CATEGORIES_AR = [
    "مطاعم هندية","مطاعم شاورما","مطاعم لبنانية",
    "مطاعم خليجية","مطاعم أسماك","مطاعم برجر","أخرى"
]

def classify_restaurant(name, address, types, timeout_s=30):
    if not OPENAI_KEY:
        return "❌ OpenAI API key missing", None
    system_msg = (
        "أنت مساعد لتصنيف أسماء المطاعم. صنّف المطعم إلى أحد التصنيفات التالية بدقة: "
        + ", ".join(CATEGORIES_AR)
        + ". أجب فقط بالكلمة العربية المطابقة بالضبط."
    )
    user_msg = f"Name: {name}\nAddress: {address}\nTypes: {types}"
    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":system_msg},
                      {"role":"user","content":user_msg}],
            max_tokens=20,
            temperature=0.0
        )
        elapsed = round(time.time()-t0,2)
        text = resp.choices[0].message.content.strip()
        if text in CATEGORIES_AR:
            return text, elapsed
        # fallback mapping
        low = text.lower()
        if "indian" in low or "هندي" in low: return "مطاعم هندية", elapsed
        if "shawarma" in low or "شاورما" in low: return "مطاعم شاورما", elapsed
        if "lebanese" in low or "لبناني" in low: return "مطاعم لبنانية", elapsed
        if "gulf" in low or "خليج" in low: return "مطاعم خليجية", elapsed
        if "fish" in low or "seafood" in low or "سمك" in low: return "مطاعم أسماك", elapsed
        if "burger" in low or "برجر" in low: return "مطاعم برجر", elapsed
        return "أخرى", elapsed
    except Exception as e:
        return f"❌ Error: {e}", None

# -------------------------
# 3️⃣ Streamlit session state
# -------------------------
if "coords" not in st.session_state: st.session_state["coords"]=None
if "restaurants" not in st.session_state: st.session_state["restaurants"]=None
if "classified" not in st.session_state: st.session_state["classified"]=[]
if "index" not in st.session_state: st.session_state["index"]=0

# -------------------------
# 4️⃣ UI
# -------------------------
st.markdown("### 1) Paste Google Maps URL")
url = st.text_input("Google Maps URL here:")
if st.button("▶️ Start — Extract Coordinates"):
    lat,lng,t = extract_coordinates(url)
    if lat is None:
        st.error(f"Could not extract coordinates (took {t}s). Paste full address-bar URL or long Google Maps link.")
    else:
        st.session_state["coords"]=(lat,lng)
        st.success(f"Coordinates: {lat},{lng} (extraction {t}s)")

st.markdown("### 2) Fetch nearby restaurants")
if st.button("➡️ Continue — Fetch Restaurants"):
    if not st.session_state["coords"]:
        st.error("Run Start first to extract coordinates.")
    else:
        lat,lng = st.session_state["coords"]
        try:
            df = fetch_restaurants(lat,lng,MAPS_KEY)
            st.session_state["restaurants"] = df
            st.session_state["classified"]=[]
            st.session_state["index"]=0
            st.success(f"Found {len(df)} restaurants (showing first 50).")
            st.dataframe(df[["name","address","rating","types"]].head(50))
        except Exception as e:
            st.error(f"Places API error: {e}")

st.markdown("### 3) Classify restaurants step-by-step")
col1,col2=st.columns([1,3])
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
                cat,t = classify_restaurant(row["name"], row["address"], row["types"])
                st.session_state["classified"].append({
                    "name":row["name"], "address":row["address"],
                    "category":cat, "map_url":row.get("map_url",""), "time_s":t
                })
                st.session_state["index"]+=1
                st.success(f"Classified {row['name']} -> {cat} (took {t}s)")
with col2:
    st.write("Progress:", st.session_state["index"], "/", len(st.session_state["restaurants"]) if st.session_state["restaurants"] is not None else 0)

if st.session_state["classified"]:
    st.subheader("Classified so far")
    st.dataframe(pd.DataFrame(st.session_state["classified"]))
