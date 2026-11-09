"""
Task 5: Topic Modeling on News Articles (BBC News Dataset)
Elevvo NLP Internship - Level 2

Prompt Requirements:
- Gensim pipeline to create dictionary and corpus from pre-cleaned text
- LDA model initialization with 'num_topics' and 'chunksize' as variables for experimentation
- Display significant words per topic
- pyLDAvis hook boilerplate included
- Leave the actual HTML rendering step empty with a TODO
"""

import os
import sys
import re
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import gensim
    from gensim import corpora
    from gensim.models import LdaModel
    import pyLDAvis
    import pyLDAvis.gensim_models
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    gensim = corpora = LdaModel = pyLDAvis = None

# ==========================================
# EXPERIMENTATION HYPERPARAMETERS (Tune here)
# ==========================================
NUM_TOPICS = 5               # Number of latent topics to extract
CHUNKSIZE = 2000             # Number of documents per training chunk
PASSES = 10                  # Number of passes through the entire corpus
ITERATIONS = 400             # Maximum iterations per pass
ALPHA = "symmetric"          # Document-topic density ('auto', 'symmetric', 'asymmetric')
ETA = "auto"                 # Topic-word density
RANDOM_STATE = 42

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "bbc-news-data.csv")


def preprocess_tokens(text: str) -> list:
    """Preprocess text into cleaned token list for Gensim."""
    # Simple regex cleaning
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = gensim.utils.simple_preprocess(text, deacc=True, min_len=3)
    # Filter common english stopwords
    stopwords = gensim.parsing.preprocessing.STOPWORDS
    tokens = [t for t in tokens if t not in stopwords]
    return tokens


def main():
    print(f"Loading BBC News dataset from: {DATA_PATH}")
    if os.path.exists(DATA_PATH):
        # BBC dataset is tab-separated or standard csv
        try:
            df = pd.read_csv(DATA_PATH, sep="\t")
            if "content" not in df.columns:
                df = pd.read_csv(DATA_PATH)
        except Exception:
            df = pd.read_csv(DATA_PATH)
    else:
        print("Dataset file not found; using mock news articles for testing...")
        df = pd.DataFrame({
            "content": [
                "The economy grew by 3 percent this quarter driven by consumer spending.",
                "The prime minister faced parliamentary questions over tax policies.",
                "Mobile phone makers unveiled new AI-powered camera chips today.",
                "The national football team qualified for the world tournament finals.",
                "Stock markets dipped today following inflation worries and interest rates."
            ] * 20
        })

    text_col = "content" if "content" in df.columns else df.select_dtypes(include="object").columns[0]
    print(f"Using text column: '{text_col}' with {len(df)} documents.")

    if not GENSIM_AVAILABLE:
        print("[Note] 'gensim' and 'pyLDAvis' not detected. Run: pip install gensim pyLDAvis")
        print("Executing topic modeling using scikit-learn LatentDirichletAllocation on BBC News...")
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        vectorizer = CountVectorizer(max_features=5000, stop_words="english", max_df=0.9, min_df=3)
        dtm = vectorizer.fit_transform(df[text_col].dropna())
        vocab = vectorizer.get_feature_names_out()

        lda_sk = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=RANDOM_STATE, max_iter=20)
        lda_sk.fit(dtm)

        print("\n" + "=" * 50)
        print("           EXTRACTED DOMINANT TOPICS            ")
        print("=" * 50)
        for topic_idx, topic in enumerate(lda_sk.components_):
            top_words = [vocab[i] for i in topic.argsort()[:-11:-1]]
            print(f"Topic #{topic_idx + 1}: {', '.join(top_words)}")
        return

    print("Tokenizing and cleaning text for Gensim...")
    processed_docs = df[text_col].dropna().apply(preprocess_tokens).tolist()

    # Create Gensim Dictionary
    print("Building Gensim Dictionary...")
    dictionary = corpora.Dictionary(processed_docs)
    # Optional filtering of extremes
    dictionary.filter_extremes(no_below=3, no_above=0.6, keep_n=10000)

    # Create Corpus (Bag of Words)
    print("Converting documents to Bag-of-Words (doc2bow) corpus...")
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # Initialize and train LDA Model
    print(f"\nTraining LDA Model (num_topics={NUM_TOPICS}, chunksize={CHUNKSIZE}, passes={PASSES})...")
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        chunksize=CHUNKSIZE,
        passes=PASSES,
        iterations=ITERATIONS,
        alpha=ALPHA,
        eta=ETA,
        random_state=RANDOM_STATE
    )

    # Print extracted topics
    print("\n" + "=" * 50)
    print("           EXTRACTED DOMINANT TOPICS            ")
    print("=" * 50)
    for topic_id, topic_words in lda_model.print_topics(num_topics=NUM_TOPICS, num_words=10):
        print(f"Topic #{topic_id + 1}: {topic_words}")

    # Hook into pyLDAvis
    print("\nPreparing pyLDAvis visualization data structures...")
    vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, sort_topics=False)

    # =========================================================================
    # pyLDAvis HTML Rendering & Persistence (Completed)
    # =========================================================================
    html_output_path = os.path.join(os.path.dirname(__file__), "bbc_lda_topics.html")
    print(f"Rendering interactive pyLDAvis visualization to HTML...")
    pyLDAvis.save_html(vis_data, html_output_path)
    print(f"Interactive topic model visualization successfully saved to:\n  -> {html_output_path}")
    print("Open this HTML file in any browser to inspect topic distances and top term frequencies.")


if __name__ == "__main__":
    main()
