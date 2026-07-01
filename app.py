import os
import io
import json
import pickle
import traceback
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
import numpy as np
import requests
from openai import OpenAI 

# ============================================================
# 🔐 ENV LOADING
# ============================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY") 
SECRET_KEY = os.getenv("SECRET_KEY", "a-default-secret-key")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CROP_MODEL_PATH_STR = os.path.join(
    BASE_DIR,
    "models",
    "crop_recommendation.pkl"
)

# Check Keys
if not GROQ_API_KEY: print("❌ ERROR: GROQ_API_KEY missing.")
if not WEATHER_API_KEY: print("❌ ERROR: WEATHER_API_KEY missing.")

# ============================================================
# ============================================================


# ============================================================
# 🤖 AI Configuration (Groq)
# ============================================================
ai_client = None
try:
    if GROQ_API_KEY:
        ai_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
        )
        print("✔ Groq AI client loaded.")
    else:
        print("❌ Groq Key missing.")
except Exception as e:
    print(f"❌ AI Error: {e}")
    ai_client = None

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = SECRET_KEY

# ============================================================
# 🌐 Translations Dictionary
# ============================================================
translations = {
    'en': {
        'app_title': 'Farmer Friendly App', 'developed_by': 'guided by', 'language': 'Language',
        'dashboard': 'Dashboard', 'crop_recommendation': 'Crop Recommendation', 'disease_detection': 'Disease Detection',
        'fertilizer_advisory': 'Fertilizer Advisory', 'ai_chatbot': 'AI Chatbot', 'weather': 'Weather',
        'crop_recommendation_desc': 'Get scientific crop suggestions based on soil.',
        'disease_detection_desc': 'Identify plant diseases instantly.',
        'fertilizer_advisory_desc': 'Get expert fertilizer and treatment recommendations.',
        'ai_chatbot_desc': 'Your 24/7 farming assistant.',
        'weather_desc': 'Live local weather updates & 15-day forecast.',
        'location_label': 'Location (village/district)', 'season_label': 'Season', 'soil_type_label': 'Soil Type',
        'irrigation_label': 'Irrigation', 'area_label': 'Field Area (hectares)', 'prev_crop_label': 'Previous Crop',
        'budget_label': 'Budget', 'market_label': 'Market preference', 'get_recommendation_btn': 'Get Recommendation',
        'top_recommendations': 'Top Recommendations', 'features_estimated': '',
        'upload_label': 'Upload leaf image', 'detect_btn': 'Detect Disease', 'get_fertilizer_btn': 'Get Fertilizer Advice',
        'disease_result': 'Disease', 'confidence_result': 'Confidence', 'healthy_result': 'The plant looks healthy!',
        'crop_label': 'Crop', 'disease_label': 'Disease (if known)', 'get_advisory_btn': 'Get Advisory',
        'city_label': 'City or use GPS', 'check_weather_btn': 'Check Weather', 'use_gps_btn': 'Use GPS',
        'weather_feels_like': 'Feels like', 'weather_humidity': 'Humidity', 'weather_wind': 'Wind',
        'forecast_title': '15-Day Forecast', 'forecast_date': 'Date', 'forecast_max': 'Max', 'forecast_min': 'Min', 'forecast_condition': 'Condition',
        'chat_label': 'Ask your question', 'chat_btn': 'Ask', 'chat_bot_label': 'Bot',
        'rice': 'Rice', 'maize': 'Maize', 'chickpea': 'Chickpea', 'Healthy': 'Healthy', 'Powdery': 'Powdery', 'Rust': 'Rust',
    },
    'hi': {
        'app_title': 'किसान फ्रेंडली ऐप', 'developed_by': 'मार्गदर्शन', 'language': 'भाषा',
        'dashboard': 'डैशबोर्ड', 'crop_recommendation': 'फसल की सिफारिश', 'disease_detection': 'रोग की पहचान',
        'fertilizer_advisory': 'उर्वरक सलाह', 'ai_chatbot': 'एआई चैटबॉट', 'weather': 'मौसम',
        'crop_recommendation_desc': 'वैज्ञानिक फसल सुझाव प्राप्त करें।',
        'disease_detection_desc': 'तुरंत पौधों के रोगों की पहचान करें।',
        'fertilizer_advisory_desc': 'विशेषज्ञ उर्वरक सिफारिशें प्राप्त करें।',
        'ai_chatbot_desc': 'आपका 24/7 कृषि सहायक।',
        'weather_desc': 'लाइव स्थानीय मौसम अपडेट और 15-दिन का पूर्वानुमान।',
        'location_label': 'स्थान (गाँव/जिला)', 'season_label': 'मौसम', 'soil_type_label': 'मिट्टी का प्रकार',
        'irrigation_label': 'सिंचाई', 'area_label': 'क्षेत्र (हेक्टेयर)', 'prev_crop_label': 'पिछली फसल',
        'budget_label': 'बजट', 'market_label': 'बाजार वरीयता', 'get_recommendation_btn': 'सिफारिश प्राप्त करें',
        'top_recommendations': 'शीर्ष सिफारिशें', 'features_estimated': '',
        'upload_label': 'पत्ती की छवि अपलोड करें', 'detect_btn': 'रोग का पता लगाएं', 'get_fertilizer_btn': 'उर्वरक सलाह',
        'disease_result': 'रोग', 'confidence_result': 'आत्मविश्वास', 'healthy_result': 'पौधा स्वस्थ है!',
        'crop_label': 'फसल', 'disease_label': 'रोग', 'get_advisory_btn': 'सलाह लें',
        'city_label': 'शहर या जीपीएस', 'check_weather_btn': 'मौसम जांचें', 'use_gps_btn': 'जीपीएस',
        'weather_feels_like': 'जैसा लगता है', 'weather_humidity': 'नमी', 'weather_wind': 'हवा',
        'forecast_title': '15-दिन का पूर्वानुमान', 'forecast_date': 'तारीख', 'forecast_max': 'अधिकतम', 'forecast_min': 'न्यूनतम', 'forecast_condition': 'स्थिति',
        'chat_label': 'प्रश्न पूछें', 'chat_btn': 'पूछें', 'chat_bot_label': 'बॉट',
        'rice': 'चावल', 'maize': 'मक्का', 'chickpea': 'चना', 'Healthy': 'स्वस्थ', 'Powdery': 'पाउडर फफूंद', 'Rust': 'जंग',
    },
    'te': {
         'app_title': 'రైతు స్నేహపూర్వక యాప్', 'developed_by': 'మార్గదర్శకత్వం', 'language': 'భాష',
         'dashboard': 'డాష్‌బోర్డ్', 'crop_recommendation': 'పంట సిఫార్సు', 'disease_detection': 'వ్యాధి గుర్తింపు',
         'fertilizer_advisory': 'ఎరువుల సలహా', 'ai_chatbot': 'AI చాట్‌బాట్', 'weather': 'వాతావరణం',
         'crop_recommendation_desc': 'శాస్త్రీయ పంట సూచనలను పొందండి.',
         'disease_detection_desc': 'మొక్కల వ్యాధులను గుర్తించండి.',
         'fertilizer_advisory_desc': 'ఎరువుల సిఫార్సులను పొందండి.',
         'ai_chatbot_desc': 'మీ 24/7 వ్యవసాయ సహాయకుడు.', 'weather_desc': 'వాతావరణ నవీకరణలు & 15-రోజుల సూచన.',
         'location_label': 'స్థానం', 'season_label': 'సీజన్', 'soil_type_label': 'నేల రకం',
         'irrigation_label': 'నీటిపారుదల', 'area_label': 'విస్తీర్ణం', 'prev_crop_label': 'మునుపటి పంట',
         'budget_label': 'బడ్జెట్', 'market_label': 'మార్కెట్ ప్రాధాన్యత', 'get_recommendation_btn': 'సిఫార్సు పొందండి',
         'top_recommendations': 'అగ్ర సిఫార్సులు', 'features_estimated': '',
         'upload_label': 'ఫోటో అప్‌లోడ్ చేయండి', 'detect_btn': 'గుర్తించండి', 'get_fertilizer_btn': 'సలహా పొందండి',
         'disease_result': 'వ్యాధి', 'confidence_result': 'విశ్వాసం', 'healthy_result': 'మొక్క ఆరోగ్యంగా ఉంది!',
         'crop_label': 'పంట', 'disease_label': 'వ్యాధి', 'get_advisory_btn': 'సలహా పొందండి',
         'city_label': 'నగరం/GPS', 'check_weather_btn': 'తనిఖీ చేయండి', 'use_gps_btn': 'GPS',
         'weather_feels_like': 'అనిపిస్తుంది', 'weather_humidity': 'తేమ', 'weather_wind': 'గాలి',
         'forecast_title': '15-రోజుల సూచన', 'forecast_date': 'తేదీ', 'forecast_max': 'గరిష్ట', 'forecast_min': 'కనిష్ట', 'forecast_condition': 'స్థితి',
         'chat_label': 'ప్రశ్న అడగండి', 'chat_btn': 'అడగండి', 'chat_bot_label': 'బాట్',
         'rice': 'బియ్యం', 'maize': 'మొక్కజొన్న', 'chickpea': 'శనగలు', 'Healthy': 'ఆరోగ్యకరమైన', 'Powdery': 'బూజు తెగులు', 'Rust': 'కుంకుమ తెగులు',
    },
    'gu': {}
}

def get_lang_data():
    lang_code = request.args.get('lang', 'en')
    if lang_code not in translations or not translations[lang_code]: lang_code = 'en'
    return translations[lang_code], lang_code

# ============================================================
# 🌱 Load Models
# ============================================================
crop_model = None
if CROP_MODEL_PATH_STR and Path(CROP_MODEL_PATH_STR).exists():
    try:
        with open(CROP_MODEL_PATH_STR, "rb") as f: crop_model = pickle.load(f)
        print(f"✔ Crop model loaded.")
    except Exception as e: print(f"❌ Crop Model Error: {e}")

disease_model = None
DISEASE_LABELS = ["Healthy", "Powdery", "Rust"]


# ============================================================
# 🛠️ Helpers (Safeguarded)
# ============================================================
def preprocess_image(pil_img, size=(224, 224)):
    img = pil_img.convert("RGB").resize(size)
    return np.expand_dims(np.array(img) / 255.0, axis=0)

def get_weather_full(city=None):
    if not WEATHER_API_KEY or not city: return None
    try:
        base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
        url = f"{base_url}/{city}?unitGroup=metric&key={WEATHER_API_KEY}&contentType=json"
        
        # Timeout added to prevent hanging
        res = requests.get(url, timeout=4)
        
        if res.status_code != 200:
            print(f"❌ Weather API Error ({res.status_code}): {res.text}")
            return None
            
        data = res.json()
        if 'days' not in data: return None
        return data
    except Exception as e: 
        print(f"⚠️ Weather Connection Issue: {e}")
        return None

def get_current_weather(city):
    data = get_weather_full(city)
    if not data: return None
    curr = data.get('currentConditions', {})
    return {
        "temperature": curr.get('temp'),
        "humidity": curr.get('humidity'),
        "rainfall_mm": curr.get('precip', 0) or 0
    }

def estimate_features(data, weather=None):
    soil, season = data.get("soil_type", "loamy").lower(), data.get("season", "kharif").lower()
    prev, budget = data.get("previous_crop", "").lower(), data.get("budget", "medium").lower()

    N, P, K = 50.0, 25.0, 25.0

    if "clay" in soil: P *= 0.9; K *= 1.1
    if "sandy" in soil: N *= 1.05; P *= 1.1; K *= 0.9
    if prev in ["groundnut", "peanut", "soybean", "legume"]: N *= 0.7
    if budget == "low": N *= 0.85; P *= 0.85; K *= 0.85
    
    ph = 6.5

    if weather:
        temp = float(weather.get("temperature", 26))
        hum = float(weather.get("humidity", 65))
        rain = 180.0 if 'kharif' in season else (40.0 if 'rabi' in season else 20.0)
    else:
        print(f"--- Simulating weather for {season} ---")
        if 'kharif' in season: temp, hum, rain = 28.0, 85.0, 200.0
        elif 'rabi' in season: temp, hum, rain = 20.0, 45.0, 40.0
        elif 'zaid' in season: temp, hum, rain = 35.0, 35.0, 10.0
        else: temp, hum, rain = 26.0, 65.0, 40.0

    import random
    N += random.uniform(-5, 5)
    temp += random.uniform(-2, 2)
    hum += random.uniform(-5, 5)

    return [int(N), int(P), int(K), round(temp, 2), round(hum, 2), round(ph, 2), round(rain, 2)]

def call_ai_model(prompt):
    if not ai_client: return "[AI Error] Client not loaded."
    try:
        response = ai_client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[
                {"role": "system", "content": "You are a helpful farming assistant. Be concise. Use Markdown."}, 
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e: return f"[Groq Error] {str(e)}"

# ============================================================
# 🛣️ Routes
# ============================================================
@app.route("/")
def dash():
    lang, code = get_lang_data()
    return render_template("dashboard.html", lang=lang, current_lang=code)

@app.route("/<page>")
def serve_page(page):
    lang, code = get_lang_data()
    if page in ["dashboard.html", "crop_recommendation.html", "disease_detection.html", 
                "fertilizer_advisory.html", "weather.html", "ai_interactive.html"]:
        return render_template(page, lang=lang, current_lang=code)
    return "Page not found", 404

@app.route("/api/predict_crop", methods=["POST"])
def predict_crop():
    try:
        lang_data, _ = get_lang_data()
        data = request.json
        
        weather = get_current_weather(data.get("city"))
        features = estimate_features(data, weather)
        
        if not crop_model: 
            return jsonify({"error": "Crop model is not loaded."}), 500
        
        prediction = crop_model.predict([features])[0]
        translated_name = lang_data.get(prediction.lower(), prediction.capitalize())
        
        

        return jsonify({
            "input_features": features,
            "recommendations": [{"crop": translated_name, "score": 1.0}],
            "weather_used": weather or "Simulated"
        })
    except Exception as e:
        print(f"❌ SERVER ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
'''
@app.route("/api/predict_disease", methods=["POST"])
def predict_disease():
    if not disease_model: return jsonify({"error": "Model not loaded"}), 500
    try:
        lang_data, _ = get_lang_data()
        img = Image.open(request.files["image"])
        arr = preprocess_image(img)
        preds = disease_model.predict(arr)[0]
        idx = int(np.argmax(preds))
        
        label_key = DISEASE_LABELS[idx]
        translated_label = lang_data.get(label_key, label_key)
        
        
        
        return jsonify({"label": translated_label, "label_key": label_key, "confidence": float(preds[idx])})
    except Exception as e:
        print(f"❌ Disease Error: {e}")
        return jsonify({"error": str(e)}), 500
'''
@app.route("/api/fertilizer_advice", methods=["POST"])
def fertilizer_advice():
    data, (lang_data, code) = request.json, get_lang_data()
    prompt = f"Act as an agronomist. Language: {code}. Crop: {data.get('crop')}. Disease: {data.get('disease')}. Soil: {data.get('soil_type')}. Suggest fertilizer (NPK) and treatment. Keep it short. Use Markdown."
    return jsonify({"recommendation": call_ai_model(prompt)})

@app.route("/api/chat", methods=["POST"])
def chatbot():
    data, (lang_data, code) = request.json, get_lang_data()
    prompt = f"""
Act as a helpful farmer assistant.
Respond ONLY in this language: {code}
**IMPORTANT RULE:** You MUST only answer questions related to farming, agriculture, crops, livestock, weather, soil, or fertilizer.
If the user asks a question *not* related to farming, politely decline in {code}.
The farmer asks: {data.get('message')}
"""
    return jsonify({"response": call_ai_model(prompt)})

@app.route("/api/weather")
def api_weather():
    q = request.args.get('q')
    data = get_weather_full(city=q)
    
    if not data: return jsonify({"error": "Failed to fetch weather"}), 500
    
    current = data.get('currentConditions', {})
    days = data.get('days', [])[:15]
    
    return jsonify({
        "name": data.get('address', q),
        "current": {
            "temp": current.get('temp'),
            "humidity": current.get('humidity'),
            "feelslike": current.get('feelslike'),
            "conditions": current.get('conditions'),
            "wind": current.get('windspeed'),
            "icon": current.get('icon')
        },
        "forecast": days 
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)