# --- FIX FOR STREAMLIT CLOUD: Ensure NLTK 'punkt' is available ---
import nltk
import os

nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)

nltk.data.path.append(nltk_data_dir)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_dir)

# ---------------------------------------------------------------

import joblib
import numpy as np
import string
from nltk.tokenize import word_tokenize
import streamlit as st

# Load the model and scaler
clf = joblib.load("models/sentence_clf.pkl")
scaler = joblib.load("models/scaler.pkl")

def featurize_sentence(sent):
    tokens = word_tokenize(sent)
    token_count = len(tokens) if len(tokens) > 0 else 1
    avg_word_len = float(np.mean([len(t) for t in tokens])) if tokens else 0.0
    punct_count = sum(1 for c in sent if c in string.punctuation)
    digit_count = sum(1 for c in sent if c.isdigit())
    capital_count = sum(1 for c in sent if c.isupper())
    nll = 0.0  # transformer NLL optional feature
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

        # LightGBM Booster returns probability for positive class directly
        proba = clf.predict(feats_s)[0]
        label = "AI-generated" if proba >= 0.5 else "Human-written"

        st.subheader("🔍 Prediction Result")
        st.write(f"**Label:** {label}")
        st.write(f"**AI Probability:** {proba:.4f}")

        st.subheader("📌 Input Text")
        st.write(text)
