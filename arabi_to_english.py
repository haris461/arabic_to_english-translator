import streamlit as st
import torch
import urllib.request
import os
from transformers import MarianMTModel, MarianTokenizer

# Set Streamlit page config
st.set_page_config(page_title="Arabic-English Translator", page_icon="🌍", layout="centered")

# Define model URL and path
model_url = "https://github.com/haris461/arabic_to_english-translator/releases/download/4.46.3/nmt_model.pkl"
model_path = "nmt_model.pkl"

# Download model if not exists
if not os.path.exists(model_path):
    st.write("Downloading model...")
    urllib.request.urlretrieve(model_url, model_path)
    st.write("Download successful!")

# Load the trained model
try:
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load tokenizer
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")

# Streamlit App UI
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #000000, #0a0a23);
            color: white;
        }
        [data-testid="stSidebar"] {
            background: rgba(30, 30, 30, 0.9);
            color: white;
        }
        .custom-title {
            color: #32CD32 !important;
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            text-shadow: 0px 0px 10px rgba(50, 205, 50, 0.8), 0px 0px 20px rgba(50, 205, 50, 0.6);
            animation: glow 1.5s infinite alternate;
        }
        .translated-text {
            background: rgba(50, 205, 50, 0.2);
            border-left: 5px solid #32CD32;
            color: white;
            font-size: 22px;
            font-weight: bold;
            padding: 15px;
            margin-top: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0px 0px 10px rgba(50, 205, 50, 0.5);
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='custom-title'>🌍 LinguaBridge (Arabic to English Translator) 📝</h1>", unsafe_allow_html=True)

# User Input
arabic_text = st.text_area("Enter Arabic text:", height=150)

# Translate Function
def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        translated_tokens = model.generate(**inputs)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

# Translate Button
if st.button("Translate 🔁"):
    if arabic_text.strip() == "":
        st.warning("⚠️ Please enter Arabic text to translate.")
    else:
        translated_text = translate(arabic_text)
        st.markdown(f"<p class='translated-text'><strong>Translated Text:</strong> {translated_text}</p>", unsafe_allow_html=True)


