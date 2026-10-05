"""
Train and compare 3 spam classifiers, then save the best full pipeline.

Run from the project root:
    python training/train_model.py
"""
import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_model.pkl")

# Columns found by inspecting the dataset:
#   v1 = label ("ham" / "spam"), v2 = SMS text.
# The 3 "Unnamed" columns only hold overflow fragments, so they are ignored.
LABEL_COLUMN = "v1"
TEXT_COLUMN = "v2"


def clean_text(text):
    """Light cleaning: collapse extra whitespace. (Lowercasing, punctuation
    removal and stop-word removal are done inside TfidfVectorizer so the
    saved pipeline can handle raw text from the web app.)"""
    return re.sub(r"\s+", " ", str(text)).strip()


def load_data():
    # The file is not UTF-8, so latin-1 is needed.
    df = pd.read_csv(DATA_PATH, encoding="latin-1", usecols=[LABEL_COLUMN, TEXT_COLUMN])
    print("Loaded:", df.shape)
    print("Labels:", df[LABEL_COLUMN].value_counts().to_dict())

    df = df.dropna()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)

    before = len(df)
    df = df.drop_duplicates(subset=[LABEL_COLUMN, TEXT_COLUMN])
    print(f"Removed {before - len(df)} duplicate rows -> {len(df)} remain")

    # Encode labels: ham = 0, spam = 1
    df["label"] = df[LABEL_COLUMN].map({"ham": 0, "spam": 1})
    assert df["label"].notna().all(), "Unexpected label values found"
    return df


def make_pipeline(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english",
                                  ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", classifier),
    ])


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df[TEXT_COLUMN], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    print(f"Train: {len(X_train)}  Test: {len(X_test)}\n")

    models = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        # LinearSVC has no probabilities, so we wrap it to get confidence scores.
        "Linear SVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=5),
    }

    results, trained = [], {}
    for name, clf in models.items():
        pipe = make_pipeline(clf)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, pred)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
        })
        trained[name] = pipe
        print(f"--- {name} ---")
        print("Confusion matrix [rows=actual, cols=predicted] (ham, spam):")
        print(cm, "\n")

    table = pd.DataFrame(results).set_index("Model")
    print("Model comparison (spam = positive class):")
    print(table.round(4), "\n")

    # Best model = highest F1 (ties broken by recall)
    best_name = table.sort_values(["F1", "Recall"], ascending=False).index[0]
    print("Best model:", best_name)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(trained[best_name], MODEL_PATH)
    print("Saved pipeline to", MODEL_PATH)


if __name__ == "__main__":
    main()
