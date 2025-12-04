# train.py - simplified prototype training script
# Builds a synthetic sentence dataset, featurizes sentences, trains a LightGBM model and saves it.

import os
import random
import joblib
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import lightgbm as lgb
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt', quiet=True)

# optionally use transformers for nll feature
use_transformers = False
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

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def make_synthetic_sentences(n_per_class=1200):
    humans = [
        "I visited my grandparents last weekend and we cooked together.",
        "The experiment's results are shown in Figure 2 and indicate strong correlation.",
        "In my opinion, we should consider a longer timeframe for the study.",
        "She noted that the dataset contained missing entries that had to be imputed.",
        "I think this novel provides an interesting perspective on the era.",
        "Yesterday I took a long walk through the market and bought fresh fruit.",
        "We discussed the problem for hours and eventually reached a compromise.",
        "The meeting was postponed due to scheduling conflicts across teams.",
        "He made several minor errors in the math, but the concept is valid.",
        "Can you bring the laptop to class tomorrow?"
    ]
    ai_templates = [
        "Artificial intelligence is revolutionizing the way we solve complex problems.",
        "This paper proposes a novel framework that significantly improves performance.",
        "In conclusion, the proposed approach leads to better generalization on benchmarks.",
        "Recent developments in language models demonstrate scalable improvements.",
        "The model was trained on large corpora and exhibits robust behavior.",
        "We explore the implications of the proposed architecture and its benefits.",
        "Empirical results suggest that the algorithm achieves state-of-the-art accuracy.",
        "This method leverages attention mechanisms to enhance representational capacity.",
        "Experiments reveal consistent improvements across multiple datasets.",
        "The findings indicate potential applications in various industrial domains."
    ]
    human_sents = []
    ai_sents = []
    for _ in range(n_per_class):
        s = random.choice(humans)
        if random.random() < 0.5:
            s = s + " " + random.choice(["It was fun.", "I remember that.", "We talked about it."])
        human_sents.append(s)
        s2 = random.choice(ai_templates)
        if random.random() < 0.4:
            s2 = s2 + " " + random.choice(["Further, we discuss implications.", "Detailed analysis follows.", "Additional experiments confirm this."])
        ai_sents.append(s2)
    return human_sents, ai_sents

def sentence_nll(sent):
    if not use_transformers or tokenizer is None or model is None:
        return 0.0
    enc = tokenizer.encode(sent, return_tensors="pt")
    if torch.cuda.is_available():
        enc = enc.to("cuda")
    with torch.no_grad():
        outputs = model(enc, labels=enc)
        return outputs.loss.item()

def featurize_sentence(sent):
    tokens = word_tokenize(sent)
    token_count = len(tokens) if len(tokens) > 0 else 1
    avg_word_len = float(np.mean([len(t) for t in tokens])) if tokens else 0.0
    punct_count = sum(1 for c in sent if c in string.punctuation)
    digit_count = sum(1 for c in sent if c.isdigit())
    capital_count = sum(1 for c in sent if c.isupper())
    nll = sentence_nll(sent)
    return {
        "token_count": token_count,
        "avg_word_len": avg_word_len,
        "punct_count": punct_count,
        "digit_count": digit_count,
        "capital_count": capital_count,
        "nll": nll
    }

def build_dataset(human_sents, ai_sents):
    rows = []
    for s in human_sents:
        f = featurize_sentence(s)
        f["label"] = 0
        f["text"] = s
        rows.append(f)
    for s in ai_sents:
        f = featurize_sentence(s)
        f["label"] = 1
        f["text"] = s
        rows.append(f)
    return pd.DataFrame(rows)

def train_and_save(df):
    features = ["token_count", "avg_word_len", "punct_count", "digit_count", "capital_count", "nll"]
    X = df[features].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lgb_train = lgb.Dataset(X_train_s, label=y_train)
    lgb_eval = lgb.Dataset(X_test_s, label=y_test, reference=lgb_train)
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05, 'seed': 42}
    print("Training LightGBM...")
    gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=200,
    valid_sets=[lgb_train, lgb_eval],
    callbacks=[
        lgb.early_stopping(stopping_rounds=30),
        lgb.log_evaluation(period=0)
    ]
)

    y_pred_proba = gbm.predict(X_test_s)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    print("\nClassification report on holdout:")
    print(classification_report(y_test, y_pred, target_names=["human", "ai"]))
    joblib.dump(gbm, os.path.join(MODEL_DIR, "sentence_clf.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"Saved model to {MODEL_DIR}/sentence_clf.pkl and scaler to {MODEL_DIR}/scaler.pkl")

if __name__ == "__main__":
    print("Building synthetic dataset...")
    human_sents, ai_sents = make_synthetic_sentences(n_per_class=1200)
    df = build_dataset(human_sents, ai_sents)
    print("Dataset size:", len(df))
    train_and_save(df)
