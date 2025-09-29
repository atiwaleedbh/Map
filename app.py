import streamlit as st
import googlemaps
from openai import OpenAI
import pandas as pd
import time
import re
from urllib.parse import urlparse, parse_qs

# ==========================
# 1️⃣ استدعاء المفاتيح من secrets
# ==========================
maps_key = st.secrets["GOOGLE_MAPS_KEY"]
openai_key = st.secrets["OPENAI_API_KEY"]

# إعداد العملاء
gmaps = googlemaps.Client(key=maps_key)
client = OpenAI(api_key=openai_key)

# ==========================
# 2️⃣ دوال مساعدة
# ==========================
def extract_coords_from_url(url):
    """
    تحويل روابط Google Maps قصيرة وطويلة إلى إحداثيات (lat,lng)
    """
    try:
        # روابط طويلة
        m = re.search(r'/@([0-9\.\-]+),([0-9\.\-]+)', url)
        if m:
            return float(m.group(1)), float(m.group(2))
        # روابط قصيرة goo.gl/maps
        parsed = urlparse(url)
        if parsed.path:
            geocode_result = gmaps.geocode(url)
            if geocode_result:
                loc = geocode_result[0]['geometry']['location']
                return loc['lat'], loc['lng']
        return None, None
    except Exception as e:
        return None, None

def fetch_nearby_places(lat, lng, radius=3000, type="restaurant"):
    """جلب قائمة الأماكن القريبة من Google Maps Places API"""
    try:
        results = gmaps.places_nearby(location=(lat, lng), radius=radius, type=type)
        places = []
        for r in results.get("results", []):
            places.append({
                "name": r.get("name"),
                "address": r.get("vicinity","")
            })
        return places
    except Exception as e:
        st.error(f"❌ Error fetching places: {e}")
        return []

def classify_place(name, description=""):
    """تصنيف المطعم باستخدام ChatGPT الجديد"""
    try:
        messages = [
            {"role": "system", "content": "You are an assistant that classifies restaurants into categories: Indian, Shawarma, Lebanese, Gulf, Seafood, Burger, Others."},
            {"role": "user", "content": f"Classify this restaurant: {name}. Description: {description}"}
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {e}"

# ==========================
# 3️⃣ واجهة Streamlit
# ==========================
st.set_page_config(page_title="Restaurant Classifier", layout="centered")
st.title("🍽️ Restaurant Classifier with ChatGPT")
st.write("Paste a Google Maps link (short or long), and classify nearby restaurants.")

# --- إدخال رابط Google Maps
map_link = st.text_input("Paste Google Maps URL here:")

if st.button("Start"):
    if map_link:
        start_time = time.time()
        lat, lng = extract_coords_from_url(map_link)
        if lat and lng:
            st.success(f"✅ Coords: {lat}, {lng} (took {time.time()-start_time:.2f}s)")

            # --- Fetch nearby restaurants
            st.info("Fetching nearby restaurants...")
            places = fetch_nearby_places(lat, lng, radius=3000)
            if not places:
                st.warning("No restaurants found.")
            else:
                st.write(f"Found {len(places)} restaurants.")
                df = pd.DataFrame(places)
                df["Category"] = ""

                # تصنيف المطاعم خطوة خطوة
                for i, row in df.iterrows():
                    st.write(f"### 🍴 {row['name']}")
                    st.write(f"Address: {row['address']}")
                    if st.button(f"Classify {row['name']}", key=f"classify_{i}"):
                        category = classify_place(row['name'], row['address'])
                        df.at[i,"Category"] = category
                        st.success(f"Classified → {category}")

                st.write("### ✅ Final Table")
                st.dataframe(df)

        else:
            st.error("❌ Could not extract coordinates from the link.")
    else:
        st.warning("⚠️ Please enter a Google Maps link first.")
