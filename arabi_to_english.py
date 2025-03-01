import streamlit as st
import torch
import urllib.request
from transformers import MarianMTModel, MarianTokenizer

# Set Streamlit page config
st.set_page_config(page_title="Arabic-English Translator", page_icon="🌍", layout="centered")

# Define model name
model_name = "Helsinki-NLP/opus-mt-ar-en"

# Load tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

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
        div.stButton > button:first-child {
            background-color: #32CD32 !important;
            color: white !important;
            border-radius: 10px !important;
            font-size: 18px !important;
            font-weight: bold !important;
            padding: 10px 20px !important;
            border: none !important;
            box-shadow: 0px 0px 10px rgba(50, 205, 50, 0.5);
        }
        div.stButton > button:first-child:hover {
            background-color: #28a745 !important;
            box-shadow: 0px 0px 15px rgba(40, 167, 69, 0.7);
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

# Footer
st.markdown("<p style='text-align:center; color:#BBBBBB; font-size:14px; margin-top:30px;'>Developed with ❤️ using Streamlit</p>", unsafe_allow_html=True)



