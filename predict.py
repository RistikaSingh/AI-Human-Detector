import sys
import joblib
import string
import numpy as np
from nltk.tokenize import word_tokenize

MODEL = "models/sentence_clf.pkl"
SCALER = "models/scaler.pkl"

def featurize_sentence(sent):
    tokens = word_tokenize(sent)
    token_count = len(tokens) if len(tokens) > 0 else 1
    avg_word_len = float(np.mean([len(t) for t in tokens])) if tokens else 0.0
    punct_count = sum(1 for c in sent if c in string.punctuation)
    digit_count = sum(1 for c in sent if c.isdigit())
    capital_count = sum(1 for c in sent if c.isupper())
    nll = 0.0
    return [token_count, avg_word_len, punct_count, digit_count, capital_count, nll]

def main():
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter sentence: ").strip()
    
    X = np.array(featurize_sentence(text)).reshape(1, -1)
    
    scaler = joblib.load(SCALER)
    model = joblib.load(MODEL)
    
    Xs = scaler.transform(X)
    
    # LightGBM Booster outputs probability directly
    p = model.predict(Xs)[0]
    
    label = "ai" if p >= 0.5 else "human"
    
    print(f"Text: {text}")
    print(f"Predicted: {label} (ai-prob={p:.4f})")

if __name__ == "__main__":
    main()
