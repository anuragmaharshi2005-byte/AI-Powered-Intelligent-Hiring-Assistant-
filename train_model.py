"""
train_model.py
---------------
Trains a resume-category classifier on data/resumes_dataset.jsonl and saves
the fitted TF-IDF vectorizer, label encoder and model to the models/ folder.

Why this model, and why this feature:
--------------------------------------
The core deliverable of this project is matching ONE resume against ONE job
description, which is fundamentally an unsupervised similarity problem
(handled in similarity.py using sentence embeddings). That part doesn't
need a trained model at all.

The "machine learning model" required by the brief is used for a
complementary, genuinely useful sub-task: predicting which job category a
resume actually belongs to, based on the 3,500 labelled resumes in
resumes_dataset.jsonl (36 categories, e.g. "Java Developer",
"Data Science", "DevOps"). This lets the app do a sanity check: if a
candidate uploads a resume and it's auto-classified as e.g. "Web Designing"
while they're applying for a "Data Science" role, that's a useful signal to
surface, in addition to the similarity score.

Model choice: Logistic Regression on TF-IDF features.
    - The classes are cleanly separable in text (technical vocabulary
      differs a lot between e.g. "Network Security Engineer" and
      "React Developer"), so a linear model on TF-IDF already performs
      well without needing anything heavier.
    - It's fast to train and re-train (seconds, not minutes), which matters
      for a project that should be easy to re-run end-to-end.
    - It's interpretable — the learned coefficients directly show which
      words push a resume toward a given category, which fits the
      "explainable" requirement of the project better than a black-box
      model would.
    - Random Forest / XGBoost were considered but don't meaningfully
      outperform Logistic Regression on sparse, high-dimensional TF-IDF
      text and are slower to train for no real benefit here.

Run this script once (`python train_model.py`) before starting the
Streamlit app; it saves the artifacts the app loads at runtime.
"""

import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from preprocessing import clean_text
from utils import CLASSIFIER_PATH, DATASET_PATH, LABEL_ENCODER_PATH, MODELS_DIR, VECTORIZER_PATH

import os


def load_dataset(path=DATASET_PATH):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading dataset...")
    df = load_dataset()
    print(f"  {len(df)} resumes, {df['Category'].nunique()} categories")

    print("Cleaning text...")
    df["clean_text"] = df["Text"].fillna("").apply(clean_text)

    # Drop any rows that ended up empty after cleaning
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Category"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], y, test_size=0.2, random_state=42, stratify=y
    )

    print("Vectorizing (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression classifier...")
    model = LogisticRegression(
        max_iter=1000,
        C=5.0,
        class_weight="balanced",
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.3f}\n")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

    joblib.dump(model, CLASSIFIER_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"Saved model artifacts to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
