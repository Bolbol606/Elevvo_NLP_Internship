# Task 2 - News Category Classifier (AG News Dataset)
# Given a news headline + description, can a model figure out whether it's about
# World news, Sports, Business, or Sci/Tech? That's a 4-class classification problem.
# We'll start with a TF-IDF + LinearSVC baseline, then tune it with GridSearchCV.

import os
import re
import glob
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# AG News labels (1-indexed in the dataset)
categories = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

data_dir = os.path.join(os.path.dirname(__file__), "data")


def load_ag_news(folder):
    """
    AG News comes as train.csv / test.csv with columns like:
    "Class Index", "Title", "Description"
    We merge all CSVs we find, standardize column names, and combine
    title + description into a single text field.
    """
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    # Read and stack all CSV files (train + test combined for full picture)
    frames = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)

    # Column names vary depending on where you got the dataset — normalize them
    rename = {}
    for col in df.columns:
        lower = col.strip().lower()
        if "class" in lower or "label" in lower:
            rename[col] = "label"
        elif "title" in lower:
            rename[col] = "title"
        elif any(k in lower for k in ["desc", "text", "content"]):
            rename[col] = "description"
    df = df.rename(columns=rename)

    # Combine title and body — more signal for the model
    if "title" in df.columns and "description" in df.columns:
        df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    elif "text" not in df.columns:
        # last resort: just use the first text column
        first_text_col = df.select_dtypes(include="object").columns[0]
        df["text"] = df[first_text_col]

    # Same for labels
    if "label" not in df.columns:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            df["label"] = df[num_cols[0]]

    return df[["text", "label"]].dropna()


def clean_text(text):
    # Pretty minimal cleaning — SVMs with TF-IDF are actually pretty robust
    # to noise, so we just drop URLs and non-letter characters.
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    # --- Load data ---
    print(f"Looking for AG News data in: {data_dir}")
    try:
        df = load_ag_news(data_dir)
        print(f"  -> Loaded {len(df):,} articles across {df['label'].nunique()} categories")
    except FileNotFoundError as e:
        # If the dataset isn't downloaded yet, we still want to show the pipeline works
        print(f"  Dataset not found ({e})")
        print("  Falling back to a tiny demo dataset so the pipeline still runs...\n")
        df = pd.DataFrame({
            "text": [
                "Oil prices hit record high amid Middle East supply fears.",
                "Manchester United secures Champions League spot with late winner.",
                "Wall Street rallies as tech earnings beat analyst expectations.",
                "NASA Mars rover discovers evidence of ancient water channels.",
            ] * 50,
            "label": [1, 2, 3, 4] * 50,
        })

    df["clean"] = df["text"].apply(clean_text)

    X = df["clean"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}\n")

    # --- Baseline: TF-IDF + LinearSVC ---
    # LinearSVC is a great first choice for text classification —
    # fast, interpretable weights, works well with high-dimensional sparse features.
    baseline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, stop_words="english")),
        ("svm",   LinearSVC(C=1.0, random_state=42, max_iter=2000)),
    ])

    print("Training baseline TF-IDF + LinearSVC...")
    baseline.fit(X_train, y_train)

    y_pred_base = baseline.predict(X_test)
    print(f"Baseline accuracy: {accuracy_score(y_test, y_pred_base) * 100:.2f}%")
    print(classification_report(y_test, y_pred_base, zero_division=0))

    # --- GridSearchCV: find better hyperparameters ---
    # We're searching over vocab size, n-gram range, and the SVM regularization strength.
    # cv=3 means 3-fold cross-validation on the training set — no test data leaks in here.
    print("Running GridSearchCV to find better hyperparameters...")
    print("(This tries 12 combinations × 3 folds = 36 fits — might take a minute)\n")

    param_grid = {
        "tfidf__max_features": [5000, 10000],
        "tfidf__ngram_range":  [(1, 1), (1, 2)],
        "svm__C":              [0.1, 1.0, 5.0],
    }

    search = GridSearchCV(
        estimator=baseline,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"\nBest parameters:  {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_ * 100:.2f}%")

    # Evaluate the winning model on the held-out test set
    best_clf = search.best_estimator_
    y_pred_best = best_clf.predict(X_test)
    print(f"Tuned test accuracy: {accuracy_score(y_test, y_pred_best) * 100:.2f}%\n")
    print(classification_report(y_test, y_pred_best, zero_division=0))


if __name__ == "__main__":
    main()
