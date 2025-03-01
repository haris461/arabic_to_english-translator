import streamlit as st
import torch
import pickle
from transformers import MarianMTModel, MarianTokenizer

# Load the trained model
with open("nmt_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load tokenizer
model_name = "Helsinki-NLP/opus-mt-ar-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Streamlit App UI
st.set_page_config(page_title="Arabic-English Translator", page_icon="🌍", layout="centered")

# Custom CSS for Enhanced Styling
st.markdown(
    """
    <style>
        /* Gradient Background */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #000000, #0a0a23);
            color: white;
        }

        /* Sidebar background color */
        [data-testid="stSidebar"] {
            background: rgba(30, 30, 30, 0.9);
            color: white;
        }

        /* Title Styling - Animated Glow Effect */
        .custom-title {
            color: #32CD32 !important; /* Parrot Green */
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            text-shadow: 0px 0px 10px rgba(50, 205, 50, 0.8), 
                         0px 0px 20px rgba(50, 205, 50, 0.6);
            animation: glow 1.5s infinite alternate;
        }

        /* Input Field Glassmorphism Effect */
        .stTextInput, .stTextArea {
            background: rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(10px);
            color: #FFFFFF !important;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 12px;
            padding: 12px;
            font-size: 16px;
            transition: 0.3s;
        }
        .stTextInput:hover, .stTextArea:hover {
            border: 1px solid #32CD32;
        }

        /* Button Styling with Animated Glow */
        .stButton>button {
            background: linear-gradient(90deg, #32CD32, #28A428);
            color: white;
            border-radius: 12px;
            padding: 14px;
            font-size: 18px;
            font-weight: bold;
            border: none;
            cursor: pointer;
            transition: 0.3s;
            box-shadow: 0px 0px 15px rgba(50, 205, 50, 0.5);
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #28A428, #32CD32);
            transform: scale(1.08);
            box-shadow: 0px 0px 20px rgba(50, 205, 50, 0.7);
        }

        /* Translated Text Styling - More Visible */
        .translated-text {
            background: rgba(50, 205, 50, 0.2); /* Light green background */
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

        /* Footer Styling */
        .footer {
            text-align: center;
            color: #BBBBBB;
            font-size: 14px;
            margin-top: 30px;
        }

        /* Animation for the glowing effect */
        @keyframes glow {
            from { text-shadow: 0px 0px 10px rgba(50, 205, 50, 0.8); }
            to { text-shadow: 0px 0px 20px rgba(50, 205, 50, 1); }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title - Now More Attractive!
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
        st.markdown(f"<p class='translated-text'>**Translated Text:** {translated_text}</p>", unsafe_allow_html=True)

# Footer
st.markdown("<p class='footer'>Developed with ❤️ using Streamlit</p>", unsafe_allow_html=True)
