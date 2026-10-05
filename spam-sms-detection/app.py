import os

import joblib
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_model.pkl")

app = Flask(__name__)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Run: python training/train_model.py")
model = joblib.load(MODEL_PATH)  # full TF-IDF + classifier pipeline


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Please enter an SMS message."}), 400

    is_spam = int(model.predict([message])[0]) == 1
    spam_probability = float(model.predict_proba([message])[0][1])
    confidence = spam_probability if is_spam else 1 - spam_probability

    return jsonify({
        "prediction": "SPAM" if is_spam else "NOT SPAM",
        "confidence": round(confidence * 100, 2),
    })


if __name__ == "__main__":
    app.run(debug=True)
