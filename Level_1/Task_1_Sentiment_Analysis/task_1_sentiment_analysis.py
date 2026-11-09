# Task 1 - Sentiment Analysis on IMDb Movie Reviews
# The idea is simple: can a model learn that "absolutely brilliant" means positive
# and "total waste of time" means negative just from the words alone?
# We'll use TF-IDF to turn text into numbers, then Logistic Regression to classify.

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --- Tune these before you run ---
# TF-IDF settings
max_vocab_size  = 10000   # how many unique words to keep (top N by frequency)
ngram_range     = (1, 2)  # unigrams + bigrams, so "not good" is one feature
min_doc_freq    = 2       # ignore words that appear in fewer than 2 reviews
max_doc_freq    = 0.95    # ignore words that appear in almost every review (too generic)
use_sublinear   = True    # dampens very common words with log scaling

# Logistic Regression settings
C               = 1.0     # regularization — lower = heavier penalty on big weights
max_iterations  = 1000    # give the solver enough room to converge
test_split      = 0.20    # 80/20 train-test split
seed            = 42

data_path = os.path.join(os.path.dirname(__file__), "data", "IMDB Dataset.csv")


def clean_review(text):
    # IMDb reviews are messy — they contain HTML tags like <br />, numbers,
    # and all sorts of punctuation. Strip all of that down to plain words.
    text = re.sub(r"<[^>]+>", " ", text)        # remove HTML tags
    text = re.sub(r"[^a-zA-Z\s]", " ", text)    # keep letters only
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)             # collapse extra spaces
    return text


def main():
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Couldn't find the dataset at:\n  {data_path}\n"
            "Make sure you ran the download script first."
        )

    print("Loading the IMDb dataset...")
    df = pd.read_csv(data_path)
    print(f"  -> {len(df)} reviews loaded. Here's a quick look:")
    print(df.head(3), "\n")

    # Clean up the review text and convert sentiment labels to 0/1
    df["clean_text"] = df["review"].astype(str).apply(clean_review)
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

    X = df["clean_text"]
    y = df["label"]

    # Split before vectorizing — the vectorizer should only see training data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=seed, stratify=y
    )
    print(f"Train: {len(X_train)} reviews | Test: {len(X_test)} reviews\n")

    # TF-IDF: turns each review into a sparse vector of word importance scores.
    # We fit only on training data so the model never "sees" test vocabulary.
    print("Vectorizing with TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features=max_vocab_size,
        ngram_range=ngram_range,
        min_df=min_doc_freq,
        max_df=max_doc_freq,
        sublinear_tf=use_sublinear,
        stop_words="english",
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec  = tfidf.transform(X_test)

    # Train the classifier
    print("Training Logistic Regression...")
    clf = LogisticRegression(C=C, max_iter=max_iterations, random_state=seed)
    clf.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    # --- Confusion Matrix ---
    # Shows exactly where the model gets confused.
    # Each cell has the raw count and what % of that true class it represents.
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    cell_labels = np.array([
        [f"{count}\n({pct:.1f}%)" for count, pct in zip(row_c, row_p)]
        for row_c, row_p in zip(cm, cm_pct)
    ])

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=cell_labels,
        fmt="",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("IMDb Sentiment — Confusion Matrix", fontsize=14, pad=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")
    plt.savefig(out_path, dpi=300)
    print(f"Confusion matrix saved -> {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
