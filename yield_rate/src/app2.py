from flask import Flask, render_template, request
import xgboost as xgb
import numpy as np
import pickle
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from deep_translator import GoogleTranslator  # ✅ Install using: pip install deep-translator

# Configuration
DEBUG_MODE = True
app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load T5 model and tokenizer
T5_MODEL_PATH = "models/t5_farming_strategy_model"
try:
    tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_PATH)
    t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_PATH).to(device)
    print("✅ T5 model loaded successfully!")
except Exception as e:
    print("❌ Error loading T5 model:", e)
    raise

# Load scaler
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
ALL_FEATURES = scaler.feature_names_in_.tolist()
FEATURE_MEANS = dict(zip(ALL_FEATURES, scaler.mean_))

# Load crop models
CROP_MODELS = {}
CROPS = ['rice', 'wheat', 'maize', 'barley', 'sorghum', 
         'soyabean', 'groundnut', 'kharif_sorghum', 'rabi_sorghum',
         'pearl_millet', 'finger_millet', 'chickpea', 'pigeonpea',
         'minor_pulses', 'safflower', 'castor', 'linseed',
         'sunflower', 'oilseeds', 'sugarcane', 'cotton']

for crop in CROPS:
    model_path = f'models/{crop}_model.bin'
    if os.path.exists(model_path):
        CROP_MODELS[crop] = xgb.Booster(model_file=model_path)

INPUT_FEATURES = ['temperature', 'humidity', 'nitrogen', 'phosphorus', 'potassium', 'rainfall']

def create_input_vector(form_data):
    """Create input vector with feature means instead of zeros."""
    try:
        features = {name: FEATURE_MEANS.get(name, np.random.uniform(0.1, 1.0)) for name in ALL_FEATURES}
        
        for feature in INPUT_FEATURES:
            if feature in form_data:
                features[feature] = float(form_data[feature])
        
        input_df = pd.DataFrame([features])[ALL_FEATURES]
        
        if scaler:
            return scaler.transform(input_df)
        return input_df.values
    except Exception as e:
        print("Input creation failed:", e)
        return None

def generate_strategy_t5(crop, yield_value, temperature, rainfall, humidity, nitrogen, potassium, phosphorus):
    """Generate farming strategy using the trained T5 model."""
    input_text = f"{nitrogen} | {phosphorus} | {potassium} | {temperature} | {humidity} | {crop} | {yield_value}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = t5_model.generate(**inputs, max_length=150)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def translate_to_tamil(text):
    """Translate English text to Tamil."""
    try:
        cleaned_text = text.replace("[SEP]", " ")  # Remove [SEP] before translation
        return GoogleTranslator(source="en", target="ta").translate(cleaned_text)
    except Exception as e:
        print("❌ Translation failed:", e)
        return "தமிழ் மொழிபெயர்ப்பு தோல்வியடைந்தது"  # "Tamil translation failed"

@app.route("/", methods=["GET", "POST"])
def index():
    selected_crop = "rice"
    predicted_yield = 0.0
    strategy_en = "No recommendation available"
    strategy_ta = "பரிந்துரை கிடைக்கவில்லை"  # Tamil Default Text

    if request.method == "POST":
        try:
            form_data = {feature: request.form.get(feature, "0") for feature in INPUT_FEATURES}
            selected_crop = request.form.get("selected_crop", "rice")

            model_input = create_input_vector(form_data)
            if model_input is None:
                raise ValueError("Input preparation failed")
            
            if selected_crop in CROP_MODELS:
                dmatrix = xgb.DMatrix(model_input)
                predicted_yield = max(0, round(float(CROP_MODELS[selected_crop].predict(dmatrix)[0]), 2))
                
                # Generate recommendation
                strategy_en = generate_strategy_t5(
                    selected_crop, predicted_yield,
                    form_data['temperature'], form_data['rainfall'], form_data['humidity'],
                    form_data['nitrogen'], form_data['potassium'], form_data['phosphorus']
                )

                # Translate strategy to Tamil
                strategy_ta = translate_to_tamil(strategy_en)

            else:
                print(f"No model found for {selected_crop}")

        except Exception as e:
            print("Processing error:", e)
            predicted_yield = 0.0
            strategy_en = "Error in processing request"
            strategy_ta = "கோரிக்கையை செயல்படுத்துவதில் பிழை"

    return render_template("index.html",
                           selected_crop=selected_crop,
                           selected_crop_yield=predicted_yield,
                           recommendation=strategy_en,
                           recommendation_tamil=strategy_ta,
                           crops=CROPS)

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=DEBUG_MODE)
