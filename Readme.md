# Elevvo NLP & Machine Learning Internship Portfolio

This repository contains the codebase for my completed tasks during the Elevvo NLP & Machine Learning Internship. The projects demonstrate a progression from foundational text preprocessing and classical machine learning algorithms to advanced, LLM-driven autonomous agents.

## 🛠 Tech Stack
**Languages:** Python, SQL  
**Data & ML:** Pandas, NumPy, Scikit-Learn, Gensim  
**AI & Orchestration:** LangChain, OpenAI/Gemini API, Hugging Face  
**Tracking & Deployment:** MLflow, Streamlit, Git  

## 🚀 Featured Projects

### 1. Sentiment Analysis on Product Reviews (Level 1)
*   **Objective:** Classify product reviews as positive or negative.
*   **Implementation:** Engineered a text-cleaning pipeline utilizing regular expressions to remove HTML tags and noise.
*   **Modeling:** Transformed raw text into numerical feature representations using `TfidfVectorizer` and trained a Logistic Regression binary classifier. 

### 2. News Category Classification (Level 1)
*   **Objective:** Categorize unstructured news articles into distinct topical categories.
*   **Implementation:** Applied NLP preprocessing techniques (tokenization, stopword removal) to standardize input data.
*   **Modeling:** Built a multiclass classification architecture using Support Vector Machines (SVM) to draw complex decision boundaries between news topics.

### 3. Topic Modeling on News Articles (Level 2)
*   **Objective:** Discover hidden themes within an unlabelled corpus of news articles.
*   **Implementation:** Formulated an unsupervised learning pipeline to process text into a Gensim dictionary and corpus.
*   **Modeling:** Deployed Latent Dirichlet Allocation (LDA) to extract dominant topics and visualize word-topic distributions based on statistical co-occurrence.

### 4. Autonomous "Text-to-SQL" Agent (Industry Level)
*   **Objective:** Build an AI agent capable of querying a relational database using natural language.
*   **Implementation:** Integrated LangChain with a local SQLite database to orchestrate tool-calling workflows.
*   **Modeling:** Engineered a self-correction logic loop, allowing the LLM to catch database execution errors, interpret the SQL traceback, and autonomously rewrite valid queries without crashing.

## 📂 Repository Structure
*   `Level_1/Task_1_Sentiment_Analysis/` - IMDb classification scripts and models.
*   `Level_1/Task_2_News_Classification/` - AG News multiclass pipeline.
*   `Level_2/Task_5_Topic_Modeling/` - BBC News LDA implementation.
*   `Level_4/Task_10_Autonomous_SQL_Agent/` - LangChain SQL agent architecture.