import streamlit as st
import torch
import os
import urllib.request
from transformers import MarianMTModel, MarianTokenizer

# Define model URL & path
MODEL_URL = "https://github.com/haris461/arabic_to_english-translator/releases/download/4.46.3/nmt_model.pth"
MODEL_PATH = "nmt_model.pth"
MODEL_NAME = "Helsinki-NLP/opus-mt-ar-en"

# Function to download model safely
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("📥 Downloading model... Please wait.")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            st.write("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Model download failed: {e}")
            return False
    return True

# Check for corrupted model file
if os.path.exists(MODEL_PATH):
    try:
        torch.load(MODEL_PATH, map_location="cpu")
    except Exception:
        st.warning("⚠️ Corrupted model file detected! Re-downloading...")
        os.remove(MODEL_PATH)
        download_model()

# Ensure model exists before proceeding
if not download_model():
    st.stop()

# Load tokenizer
try:
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
except Exception as e:
    st.error(f"❌ Failed to load tokenizer: {e}")
    st.stop()

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
if not model:
    st.stop()

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
