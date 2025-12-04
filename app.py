# app.py - simplified Flask app for sentence-level classification
import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt', quiet=True)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "sentence_clf.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

app = Flask(__name__)

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError("Model not found. Run train.py first to create models/sentence_clf.pkl and scaler.pkl")

clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# optional scoring model
use_transformers = True
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
except Exception:
    use_transformers = False
    tokenizer = None
    model = None

def sentence_nll(sent):
    if not use_transformers or tokenizer is None or model is None:
        return 0.0
    enc = tokenizer.encode(sent, return_tensors="pt")
    if torch.cuda.is_available():
        enc = enc.to("cuda")
    with torch.no_grad():
        outputs = model(enc, labels=enc)
        return outputs.loss.item()

def featurize_sentence_simple(sent):
    tokens = word_tokenize(sent)
    token_count = len(tokens) if len(tokens)>0 else 1
    avg_word_len = float(np.mean([len(t) for t in tokens])) if tokens else 0.0
    punct_count = sum(1 for c in sent if c in ".:,;!?()"'-")
    digit_count = sum(1 for c in sent if c.isdigit())
    capital_count = sum(1 for c in sent if c.isupper())
    nll = sentence_nll(sent)
    return [token_count, avg_word_len, punct_count, digit_count, capital_count, nll]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/classify", methods=["POST"])
def classify():
    data = request.json
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error":"empty text"}), 400
    sents = sent_tokenize(text)
    features = [featurize_sentence_simple(s) for s in sents]
    X = np.array(features, dtype=float)
    Xs = scaler.transform(X)
    probs = clf.predict(Xs) if hasattr(clf, "predict") and not hasattr(clf, "predict_proba") else clf.predict_proba(Xs)[:,1]
    if probs.dtype == int or probs.dtype == bool:
        probs = probs.astype(float)
    out = [{"sentence": s, "ai_prob": float(p)} for s,p in zip(sents, probs)]
    mean_prob = float(np.mean(probs)) if len(probs)>0 else 0.0
    ai_fraction = float(np.mean(probs >= 0.5))
    if ai_fraction > 0.5:
        doc_label = "AI"
    elif ai_fraction > 0.1:
        doc_label = "HYBRID"
    else:
        doc_label = "HUMAN"
    return jsonify({"sentences": out, "doc_label": doc_label, "mean_prob": mean_prob, "ai_fraction": ai_fraction})

if __name__ == "__main__":
    app.run(debug=True)
