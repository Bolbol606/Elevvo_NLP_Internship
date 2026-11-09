"""
Task 1: Sentiment Analysis on Product Reviews (IMDb Reviews)
Elevvo NLP Internship - Level 1

Prompt Requirements:
- TF-IDF Vectorization with Logistic Regression
- Explicit hyperparameters at the top for manual tuning
- accuracy_score evaluation
- TODO placeholder for custom confusion matrix visualization
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# HYPERPARAMETERS & CONFIGURATION (Tune here)
# ==========================================
MAX_FEATURES = 10000          # Maximum vocabulary size for TF-IDF
NGRAM_RANGE = (1, 2)          # Unigrams and bigrams
MIN_DF = 2                    # Minimum document frequency
MAX_DF = 0.95                 # Maximum document frequency
SUB_LINEAR_TF = True          # Apply sublinear scaling (1 + log(tf))

# Logistic Regression Hyperparameters
C_PARAM = 1.0                 # Regularization strength (smaller = stronger)
PENALTY = "l2"                # Norm of penalty ('l1', 'l2', 'elasticnet')
SOLVER = "lbfgs"              # Optimization algorithm
MAX_ITER = 1000               # Convergence iterations
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Dataset Path
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "IMDB Dataset.csv")


def clean_text(text: str) -> str:
    """Basic text preprocessing: strip HTML tags, non-alphabetics, and lowercasing."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def main():
    print(f"Loading IMDb dataset from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please verify the data directory.")

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded successfully with shape: {df.shape}")
    print(df.head(3))

    # Basic cleaning
    print("Preprocessing review texts...")
    df["cleaned_review"] = df["review"].astype(str).apply(clean_text)
    df["target"] = df["sentiment"].map({"positive": 1, "negative": 0})

    X = df["cleaned_review"]
    y = df["target"]

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train split size: {len(X_train)} | Test split size: {len(X_test)}")

    # Vectorization
    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        sublinear_tf=SUB_LINEAR_TF,
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Initialize and train Logistic Regression model
    print("Initializing & training Logistic Regression classifier...")
    model = LogisticRegression(
        C=C_PARAM,
        penalty=PENALTY,
        solver=SOLVER,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_vec, y_train)

    # Predictions & Accuracy
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n==========================================")
    print(f"Test Accuracy Score: {acc * 100:.2f}%")
    print(f"==========================================\n")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["negative", "positive"]))

    # =========================================================================
    # Custom Confusion Matrix Visualization (Completed)
    # =========================================================================
    print("Generating custom Confusion Matrix visualization...")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Annotated labels with counts and percentages
    labels = np.array([
        [f"{count}\n({pct:.1f}%)" for count, pct in zip(row_count, row_pct)]
        for row_count, row_pct in zip(cm, cm_percent)
    ])

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        cbar=True
    )
    plt.title("IMDb Sentiment Analysis - Confusion Matrix", fontsize=14, pad=12)
    plt.xlabel("Predicted Sentiment", fontsize=12)
    plt.ylabel("Actual Sentiment", fontsize=12)
    plt.tight_layout()

    output_fig_path = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")
    plt.savefig(output_fig_path, dpi=300)
    print(f"Confusion Matrix saved to: {output_fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
