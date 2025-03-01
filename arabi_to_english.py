import streamlit as st
import torch
import pickle
import urllib.request
import os
import asyncio
from transformers import MarianMTModel, MarianTokenizer

# Fix asyncio event loop issue in Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

# Define model URL and path
model_url = "https://github.com/haris461/arabic_to_english-translator/releases/download/4.46.3/nmt_model.pkl"
model_path = "nmt_model.pkl"

# Check if model exists, otherwise download it
if not os.path.exists(model_path):
    st.write("Downloading model...")
    urllib.request.urlretrieve(model_url, model_path)
    st.write("Download successful!")

# Load the trained model
with open(model_path, "rb") as f:
    try:
        model = torch.load(f, map_location=torch.device("cpu"))
    except Exception:
        model = pickle.load(f)

model.eval()  # Ensure the model is in evaluation mode

# Load tokenizer
model_name = "Helsinki-NLP/opus-mt-ar-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Streamlit App UI
st.set_page_config(page_title="Arabic-English Translator", page_icon="🌍", layout="centered")

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

# Translation Function
def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Ensure compatibility with latest transformers
    with torch.no_grad():
        translated_tokens = model.generate(**inputs)
    
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

# Translate Button
if st.button("Translate 🔁"):
    if arabic_text.strip() == "":
        st.warning("⚠️ Please enter Arabic text to translate.")
    else:
        translated_text = translate(arabic_text)
        st.markdown(f"<p class='translated-text'><strong>Translated Text:</strong> {translated_text}</p>", unsafe_allow_html=True)

# Footer
st.markdown("<p style='text-align:center; color:#BBBBBB; font-size:14px; margin-top:30px;'>Developed with ❤️ using Streamlit</p>", unsafe_allow_html=True)

