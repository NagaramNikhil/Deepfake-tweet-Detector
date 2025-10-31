from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModel
import torch
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # ✅ FIX: Use non-GUI backend
import matplotlib.pyplot as plt
import io, base64, nltk, string, re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import shap
from lime.lime_text import LimeTextExplainer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Ensure NLTK data is available
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# ---------------------- MODEL ----------------------
MODEL_NAME = "roberta-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
transformer = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
transformer.eval()
STOPWORDS = set(stopwords.words('english'))


# ---------------------- EMBEDDING ----------------------
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out = transformer(**inputs)
    emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return emb


# ---------------------- STYLOMETRIC FEATURES ----------------------
def stylometric_features(text):
    tokens = word_tokenize(text)
    words = [w for w in tokens if any(c.isalnum() for c in w)]
    num_words = len(words)
    num_chars = len(text)
    num_sentences = len(sent_tokenize(text)) or 1

    avg_word_len = sum(len(w) for w in words) / num_words if num_words else 0
    unique_words = len(set(w.lower() for w in words))
    type_token_ratio = unique_words / num_words if num_words else 0

    stop_count = sum(1 for w in words if w.lower() in STOPWORDS)
    stop_ratio = stop_count / num_words if num_words else 0

    punct_counts = Counter(c for c in text if c in string.punctuation)
    punctuation_total = sum(punct_counts.values())
    punctuation_density = punctuation_total / num_chars if num_chars else 0
    uppercase_ratio = sum(1 for c in text if c.isupper()) / num_chars if num_chars else 0
    capitalized_words = sum(1 for w in words if w[0].isupper())

    words_per_sentence = num_words / num_sentences
    avg_sentence_len = sum(len(s.split()) for s in sent_tokenize(text)) / num_sentences

    digit_count = sum(1 for c in text if c.isdigit())
    hashtags = len(re.findall(r"#\w+", text))
    mentions = len(re.findall(r"@\w+", text))
    urls = len(re.findall(r"http[s]?://\S+", text))

    syllable_count = sum(len(re.findall(r"[aeiouyAEIOUY]+", w)) for w in words)
    flesch_reading_ease = 206.835 - 1.015 * words_per_sentence - 84.6 * (syllable_count / num_words) if num_words else 0
    flesch_kincaid_grade = 0.39 * words_per_sentence + 11.8 * (syllable_count / num_words) - 15.59 if num_words else 0

    return {
        "num_words": num_words,
        "unique_words": unique_words,
        "vocab_richness": round(type_token_ratio, 3),
        "avg_word_len": round(avg_word_len, 3),
        "num_sentences": num_sentences,
        "words_per_sentence": round(words_per_sentence, 3),
        "avg_sentence_len": round(avg_sentence_len, 3),
        "stopword_count": stop_count,
        "stopword_ratio": round(stop_ratio, 3),
        "punctuation_count": punctuation_total,
        "punctuation_density": round(punctuation_density, 3),
        "uppercase_ratio": round(uppercase_ratio, 3),
        "capitalized_words": capitalized_words,
        "digit_count": digit_count,
        "hashtags": hashtags,
        "mentions": mentions,
        "urls": urls,
        "flesch_reading_ease": round(flesch_reading_ease, 2),
        "flesch_kincaid_grade": round(flesch_kincaid_grade, 2)
    }


# ---------------------- TRAIN DUMMY CLASSIFIER ----------------------
CLASSIFIER = None
SCALER = None
BACKGROUND = None

def train_dummy():
    global CLASSIFIER, SCALER, BACKGROUND
    examples = [
        ("This is human like tweet about weather", 0),
        ("Buy cheap now click here", 1),
        ("I went to the market and bought groceries", 0),
        ("Win money now visit fake dot com", 1)
    ]
    texts = [t for t, _ in examples]
    y = np.array([lab for _, lab in examples])
    X = np.vstack([get_embedding(t) for t in texts])
    SCALER = StandardScaler().fit(X)
    Xs = SCALER.transform(X)
    clf = LogisticRegression()
    clf.fit(Xs, y)
    CLASSIFIER = clf
    BACKGROUND = np.mean(Xs, axis=0).reshape(1, -1)
    print("✅ Dummy classifier trained")

train_dummy()

LIME_EXPLAINER = LimeTextExplainer(class_names=["HUMAN", "MACHINE"])


# ---------------------- PREDICT ----------------------
def predict_from_text(text):
    emb = get_embedding(text)
    emb_s = SCALER.transform(emb.reshape(1, -1))
    proba = CLASSIFIER.predict_proba(emb_s)[0]
    pred = int(CLASSIFIER.predict(emb_s)[0])
    return pred, float(proba[pred]), emb, emb_s


# ---------------------- SHAP PLOT ----------------------
def shap_bar_plot(emb_s, topk=10):
    def f(x): return CLASSIFIER.predict_proba(x)[:, 1]
    expl = shap.KernelExplainer(f, BACKGROUND)
    shap_vals = expl.shap_values(emb_s, nsamples=50)
    vals = np.array(shap_vals[0])
    idx = np.argsort(np.abs(vals))[::-1][:topk]
    labels = [f"Feature {int(i)}" for i in idx]
    scores = vals[idx]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(labels, scores, color="#1f77b4")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), vals


# ---------------------- LIME PLOT (FIXED) ----------------------
def lime_plot(text, pred_label=0):
    def predict_proba_texts(texts):
        arr = []
        for t in texts:
            emb = get_embedding(t)
            emb_s = SCALER.transform(emb.reshape(1, -1))
            arr.append(CLASSIFIER.predict_proba(emb_s)[0])
        return np.array(arr)

    exp = LIME_EXPLAINER.explain_instance(
        text,
        predict_proba_texts,
        num_features=6,
        num_samples=300,
        labels=[0, 1]
    )

    # ✅ Ensure correct label is displayed
    fig = exp.as_pyplot_figure(label=pred_label)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    lime_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return lime_b64, exp.as_list(label=pred_label)


# ---------------------- ROUTES ----------------------
@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    tweet = data.get("tweet", "").strip()
    if not tweet:
        return jsonify({"error": "Please enter tweet text."})

    pred, conf, emb, emb_s = predict_from_text(tweet)
    label = "AI/MACHINE GENERATED" if pred == 1 else "HUMAN GENERATED"

    shap_img_b64, shap_vals, lime_img_b64, lime_list = None, [], None, []

    try:
        shap_img_b64, shap_vals = shap_bar_plot(emb_s)
    except Exception as e:
        print("SHAP error:", e)

    try:
        lime_img_b64, lime_list = lime_plot(tweet, pred_label=pred)
    except Exception as e:
        print("LIME error:", e)

    styl = stylometric_features(tweet)
    styl_text = f"Stylometric summary: {styl}"

    top_shap = []
    if len(shap_vals):
        vals = np.array(shap_vals)
        idxs = np.argsort(np.abs(vals))[::-1][:5]
        for i in idxs:
            top_shap.append({"feature": int(i), "shap": float(vals[i])})

    return jsonify({
        "input_tweet": tweet,
        "prediction": label,
        "confidence": round(conf * 100, 4),
        "stylometry": styl,
        "stylometry_text": styl_text,
        "shap_plot": shap_img_b64,
        "shap_values": top_shap,
        "lime_plot": lime_img_b64,
        "lime_explanation": lime_list
    })


# ---------------------- MAIN ----------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
