import os
import torch
import pickle
import requests
from transformers import MarianMTModel, MarianTokenizer

# Download model if not available
model_url = "https://github.com/haris461/arabic_to_english-translator/releases/download/4.46.3/nmt_model.pkl"
model_path = "nmt_model.pkl"

if not os.path.exists(model_path):
    print("Downloading model...")
    response = requests.get(model_url, stream=True)
    with open(model_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    print("Model downloaded.")

# Load Model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load Tokenizer
model_name = "Helsinki-NLP/opus-mt-ar-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
