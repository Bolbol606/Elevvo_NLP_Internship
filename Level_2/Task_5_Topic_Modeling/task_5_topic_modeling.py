# Task 5 - Topic Modeling on BBC News (LDA with Gensim)
# LDA (Latent Dirichlet Allocation) treats every document as a mixture of hidden "topics",
# and every topic as a probability distribution over words.
# Nobody tells it what the topics are — it figures them out purely from word co-occurrence patterns.
# We're going to point it at BBC News articles and see what themes it discovers.

import os
import sys
import re
import pandas as pd

# Windows terminals can choke on Unicode characters in some library outputs
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import gensim
    import gensim.corpora as corpora
    from gensim.models import LdaModel
    import pyLDAvis
    import pyLDAvis.gensim_models
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False


# --- Parameters to experiment with ---
# These are the main knobs that affect the quality of discovered topics.
num_topics  = 5      # how many latent themes to look for
chunksize   = 2000   # documents processed per training chunk — bigger = faster but more memory
passes      = 10     # how many times to loop through the whole corpus
iterations  = 400    # max EM iterations per pass
alpha       = "symmetric"  # prior on document-topic distribution (try "auto" too)
eta         = "auto"       # prior on topic-word distribution
seed        = 42

data_path = os.path.join(os.path.dirname(__file__), "data", "bbc-news-data.csv")


def tokenize(text):
    # Clean and tokenize a single document into a list of meaningful words.
    # Gensim's simple_preprocess handles lowercasing and strips punctuation.
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    if HAS_GENSIM:
        tokens = gensim.utils.simple_preprocess(text, deacc=True, min_len=3)
        # Drop stopwords (Gensim bundles a solid English stopword list)
        stopwords = gensim.parsing.preprocessing.STOPWORDS
        return [w for w in tokens if w not in stopwords]
    else:
        # Simple fallback if gensim isn't around
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return [w for w in text.split() if w not in ENGLISH_STOP_WORDS and len(w) > 2]


def run_sklearn_lda(texts):
    """
    Fallback when Gensim isn't installed.
    sklearn's LDA doesn't produce pyLDAvis output but it'll still show us topics.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    print("  (Using sklearn LDA since gensim isn't installed — pip install gensim for the full pipeline)\n")

    vec = CountVectorizer(max_features=5000, stop_words="english", max_df=0.9, min_df=3)
    dtm = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out()

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=seed, max_iter=20)
    lda.fit(dtm)

    print(f"Top words per topic (num_topics={num_topics}):\n")
    for i, topic_weights in enumerate(lda.components_):
        top_words = [vocab[j] for j in topic_weights.argsort()[:-11:-1]]
        print(f"  Topic {i+1}: {', '.join(top_words)}")


def main():
    # --- Load the BBC News dataset ---
    print(f"Loading BBC News from: {data_path}")
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path, sep="\t")
            if "content" not in df.columns:
                df = pd.read_csv(data_path)
        except Exception:
            df = pd.read_csv(data_path)
    else:
        print("Dataset not found — using a small mock corpus to show the pipeline structure.")
        df = pd.DataFrame({
            "content": [
                "The economy grew this quarter driven by consumer spending and exports.",
                "The prime minister faced parliamentary questions over tax and spending.",
                "Mobile phone makers unveiled new AI camera chips at the annual expo.",
                "The national football squad qualified for the world tournament finals.",
                "Stock markets fell on inflation worries and rising interest rates.",
            ] * 20
        })

    # Figure out which column holds the article text
    text_col = "content" if "content" in df.columns else df.select_dtypes("object").columns[0]
    print(f"  -> {len(df)} documents found in column '{text_col}'\n")

    raw_texts = df[text_col].dropna().tolist()

    # --- If gensim isn't available, run the sklearn fallback and exit early ---
    if not HAS_GENSIM:
        run_sklearn_lda(raw_texts)
        return

    # --- Tokenize ---
    print("Tokenizing documents...")
    tokenized = [tokenize(doc) for doc in raw_texts]

    # --- Build Gensim vocabulary (Dictionary) ---
    # The Dictionary maps every unique token to an integer ID.
    # We filter out very rare words (noise) and very common words (not informative).
    print("Building vocabulary dictionary...")
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=3, no_above=0.6, keep_n=10000)
    print(f"  -> Vocabulary size after filtering: {len(dictionary)} unique tokens")

    # --- Convert each document to Bag-of-Words ---
    # doc2bow gives us (word_id, count) pairs per document.
    # LDA doesn't care about word order — just counts.
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    # --- Train LDA ---
    # This is the core step. The model iteratively adjusts topic-word and
    # document-topic distributions until they stabilize (or we hit max iterations).
    print(f"\nTraining LDA  (topics={num_topics}, passes={passes}, chunksize={chunksize})...")
    print("  This can take a minute on 2000+ articles...\n")

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        chunksize=chunksize,
        passes=passes,
        iterations=iterations,
        alpha=alpha,
        eta=eta,
        random_state=seed,
    )

    # --- Print topics ---
    print("Discovered topics (top 10 words each):\n")
    for topic_id, word_weights in lda.print_topics(num_topics=num_topics, num_words=10):
        print(f"  Topic {topic_id + 1}: {word_weights}")

    # --- Save interactive visualization ---
    # pyLDAvis creates a browser-based chart where you can explore
    # topic distances and the most relevant terms for each topic.
    print("\nBuilding pyLDAvis visualization...")
    vis = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary, sort_topics=False)

    out_html = os.path.join(os.path.dirname(__file__), "bbc_lda_topics.html")
    pyLDAvis.save_html(vis, out_html)
    print(f"  -> Saved to: {out_html}")
    print("  Open this file in a browser to explore the topic landscape interactively.")


if __name__ == "__main__":
    main()
