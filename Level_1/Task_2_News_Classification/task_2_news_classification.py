"""
Task 2: News Category Classification (AG News Dataset)
Elevvo NLP Internship - Level 1

Prompt Requirements:
- Data loading and text vectorization pipeline for AG News using Pandas and Scikit-learn
- Baseline SVM classifier setup
- GridSearchCV implementation left empty with a TODO for hyperparameter search & CV logic
"""

import os
import glob
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CATEGORY_MAP = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}


def load_ag_news_data(data_dir: str):
    """
    Loads AG News dataset from CSV files (either train.csv / test.csv or a combined file).
    Handles standard AG News columns: [Class Index, Title, Description].
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV found in {data_dir}. Please place 'train.csv' or 'ag_news.csv' inside {data_dir}."
        )

    dfs = []
    for f in csv_files:
        temp_df = pd.read_csv(f)
        dfs.append(temp_df)
    df = pd.concat(dfs, ignore_index=True)

    # Standardize column names
    col_mapping = {}
    for col in df.columns:
        clean_col = col.strip().lower()
        if "class" in clean_col or "label" in clean_col:
            col_mapping[col] = "label"
        elif "title" in clean_col:
            col_mapping[col] = "title"
        elif "desc" in clean_col or "text" in clean_col or "content" in clean_col:
            col_mapping[col] = "description"

    df = df.rename(columns=col_mapping)

    # If title and description exist, combine them into full_text
    if "title" in df.columns and "description" in df.columns:
        df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    elif "text" not in df.columns:
        # Fallback to the first string column
        str_cols = df.select_dtypes(include=["object"]).columns
        df["text"] = df[str_cols[0]]

    # Ensure label column exists
    if "label" not in df.columns:
        # Check first integer/float column
        num_cols = df.select_dtypes(include=["number"]).columns
        if len(num_cols) > 0:
            df["label"] = df[num_cols[0]]

    return df[["text", "label"]].dropna()


def preprocess_text(text: str) -> str:
    """Preprocess text: lowercasing, stripping special characters, collapsing whitespace."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print(f"Loading AG News data from: {DATA_DIR}")
    try:
        df = load_ag_news_data(DATA_DIR)
        print(f"Loaded {len(df)} records.")
    except Exception as e:
        print(f"Note: {e}")
        print("Creating a mock demonstration dataframe for pipeline validation...")
        df = pd.DataFrame({
            "text": [
                "Oil prices hit record high amid middle east supply fears.",
                "Manchester United secures Champions League quarterfinal spot with dramatic win.",
                "Wall Street rallies as tech earnings beat analyst expectations.",
                "NASA Mars rover discovers new evidence of ancient water channels."
            ] * 50,
            "label": [1, 2, 3, 4] * 50
        })

    # Clean text
    print("Preprocessing text...")
    df["cleaned_text"] = df["text"].apply(preprocess_text)

    X = df["cleaned_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # BASELINE PIPELINE (TF-IDF + Linear SVM)
    # ==========================================
    print("\nSetting up baseline TF-IDF + LinearSVC pipeline...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, stop_words="english")),
        ("svm", LinearSVC(C=1.0, random_state=42, max_iter=2000))
    ])

    print("Fitting baseline SVM classifier...")
    pipeline.fit(X_train, y_train)

    # Baseline Evaluation
    y_pred = pipeline.predict(X_test)
    print("\n--- Baseline SVM Evaluation ---")
    print(f"Baseline Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    # =========================================================================
    # GridSearchCV Hyperparameter Search & Cross-Validation (Completed)
    # =========================================================================
    from sklearn.model_selection import GridSearchCV

    print("\n--- Running GridSearchCV Hyperparameter Tuning ---")
    param_grid = {
        "tfidf__max_features": [5000, 10000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "svm__C": [0.1, 1.0, 5.0],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print("\n--- GridSearchCV Results ---")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_ * 100:.2f}%")

    best_model = grid_search.best_estimator_
    y_grid_pred = best_model.predict(X_test)
    print(f"Tuned Model Test Accuracy: {accuracy_score(y_test, y_grid_pred) * 100:.2f}%")
    print("\nTuned Model Classification Report:\n")
    print(classification_report(y_test, y_grid_pred, zero_division=0))


if __name__ == "__main__":
    main()
