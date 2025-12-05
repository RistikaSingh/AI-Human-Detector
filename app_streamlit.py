# --- FIX FOR STREAMLIT CLOUD: Ensure NLTK punkt + punkt_tab are available ---
import nltk
import os

# Create a local nltk_data folder inside the app directory (Streamlit Cloud friendly)
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Add this directory to NLTK search path
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_dir)

# Ensure both punkt and punkt_tab exist (these are required by newer NLTK)
for resource in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
# ------------------------------------------------------------------------------

import joblib
import numpy as np
import string
from nltk.tokenize import word_tokenize
import streamlit as st

# Load the model and scaler
clf = joblib.load("models/sentence_clf.pkl")
scaler = joblib.load("models/scaler.pkl")

# Feature extraction function
def featurize_sentence(sent):
    tokens = word_tokenize(sent)
    token_count = len(tokens) if len(tokens) > 0 else 1
    avg_word_len = float(np.mean([len(t) for t in tokens])) if tokens else 0.0
    punct_count = sum(1 for c in sent if c in string.punctuation)
    digit_count = sum(1 for c in sent if c.isdigit())
    capital_count = sum(1 for c in sent if c.isupper())
    nll = 0.0  # placeholder, transformer option removed for Streamlit Cloud
    return [token_count, avg_word_len, punct_count, digit_count, capital_count, nll]

# ----- Streamlit UI -----

st.title("📝 AI vs Human Text Detector")
st.write("Paste a text sample below and click **Predict**.")

text = st.text_area("Enter text here:", height=200)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        feats = np.array([featurize_sentence(text)])
        feats_s = scaler.transform(feats)

        # LightGBM Booster returns probability directly
        proba = clf.predict(feats_s)[0]
        label = "AI-generated" if proba >= 0.5 else "Human-written"

        st.subheader("🔍 Prediction Result")
        st.write(f"**Label:** {label}")
        st.write(f"**AI Probability:** {proba:.4f}")

        st.subheader("📌 Input Text")
        st.write(text)

       
