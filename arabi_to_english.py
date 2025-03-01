import streamlit as st
import torch
import os
import requests  # Use requests instead of urllib
import asyncio
from transformers import MarianMTModel, MarianTokenizer

# Fix asyncio event loop issue
try:
    asyncio.set_event_loop(asyncio.new_event_loop())
except RuntimeError:
    pass

# Define model URL & path
MODEL_URL = "https://github.com/haris461/arabic_to_english-translator/releases/download/4.46.3/nmt_model.pth"
MODEL_PATH = "nmt_model.pth"

# Function to download model safely using requests
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("📥 Downloading model... Please wait.")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}  # Avoid blocking
            response = requests.get(MODEL_URL, headers=headers, stream=True)
            response.raise_for_status()  # Raise error for failed requests
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.write("✅ Model downloaded successfully!")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Model download failed: {e}")

# Remove old model files if corrupted
if os.path.exists(MODEL_PATH):
    try:
        torch.load(MODEL_PATH, map_location="cpu")
    except:
        st.warning("⚠️ Corrupted model file detected! Re-downloading...")
        os.remove(MODEL_PATH)

# Ensure model exists
download_model()

# Load tokenizer
MODEL_NAME = "Helsinki-NLP/opus-mt-ar-en"
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)

# Function to load model correctly
def load_model():
    try:
        model = MarianMTModel.from_pretrained(MODEL_NAME)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

# Load Model
model = load_model()

# Streamlit UI
st.set_page_config(page_title="Arabic-English Translator", page_icon="🌍")

st.title("🌍 Arabic to English Translator")

# User input
arabic_text = st.text_area("Enter Arabic text:", height=150)

# Translation Function
def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        translated_tokens = model.generate(**inputs)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

# Translate Button
if st.button("Translate 🔁"):
    if arabic_text.strip():
        translated_text = translate(arabic_text)
        st.success(f"**Translated Text:** {translated_text}")
    else:
        st.warning("⚠️ Please enter Arabic text.")

# Footer
st.markdown("<p style='text-align:center; color:gray;'>Developed with ❤️ using Streamlit</p>", unsafe_allow_html=True)
