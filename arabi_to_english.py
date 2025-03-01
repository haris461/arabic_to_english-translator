import streamlit as st
import torch
import urllib.request
import os
from transformers import MarianMTModel, MarianTokenizer

# Define model URL and path
MODEL_URL = "https://github.com/haris461/arabic_to_english-translator/releases/download/4.46.3/nmt_model.pkl"
MODEL_PATH = "nmt_model.pkl"

# Function to check and download model if not exists
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("📥 Downloading model... Please wait.")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.write("✅ Model downloaded successfully!")

# Download model if not present
download_model()

# Load tokenizer
MODEL_NAME = "Helsinki-NLP/opus-mt-ar-en"
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)

# Function to load model safely
def load_model():
    try:
        model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        model.eval()
        return model
    except Exception as e:
        st.warning(f"⚠️ Model loading failed: {e}")
        try:
            model = MarianMTModel.from_pretrained(MODEL_NAME)
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            model.eval()
            return model
        except Exception as e:
            st.error(f"❌ Error loading the model: {e}. Re-downloading...")
            os.remove(MODEL_PATH)
            download_model()
            return load_model()

# Load the model
model = load_model()

# Streamlit UI Configuration
st.set_page_config(page_title="Arabic-English Translator", page_icon="🌍", layout="centered")

st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #000000, #0a0a23);
            color: white;
        }
        .custom-title {
            color: #32CD32;
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            text-shadow: 0px 0px 10px rgba(50, 205, 50, 0.8);
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
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='custom-title'>🌍 Arabic to English Translator 📝</h1>", unsafe_allow_html=True)

# User Input for Arabic Text
arabic_text = st.text_area("Enter Arabic text:", height=150)

# Translation Function
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

# Footer
st.markdown("<p style='text-align:center; color:#BBBBBB; font-size:14px; margin-top:30px;'>Developed with ❤️ using Streamlit</p>", unsafe_allow_html=True)
