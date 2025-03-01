import os
import urllib.requests

MODEL_URL = "https://github.com/haris461/arabic_to_english-translator/releases/download/4.46.3/nmt_model.pkl"
MODEL_PATH = "nmt_model.pkl"

# Check if the model file exists, if not, download it
if not os.path.exists(MODEL_PATH):
    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Model downloaded successfully!")

