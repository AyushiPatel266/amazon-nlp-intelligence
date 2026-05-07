# Amazon Product Intelligence System

An end-to-end NLP system that extracts actionable business insights from 567,070 Amazon customer reviews. Built without predefined categories, the system discovers hidden themes, enables natural language search, generates automatic summaries, and classifies sentiment with two models of increasing complexity.

---

## Project Overview

Most companies collect millions of reviews but analyze very little of it. This project builds a three-component intelligence system that makes that data useful at scale.

| Component | Approach | Key Result |
|---|---|---|
| Topic Modeling | BERTopic (UMAP + HDBSCAN) | 9 distinct product themes discovered |
| Semantic Search + Summarization | FAISS + BART | Natural language search over 100k reviews |
| Sentiment Classification | TF-IDF + LR vs DistilBERT | AUC 0.9432 vs 0.9518, 200x speed trade-off |

---

## Demo
![App Screenshot](assets/screenshot1.png)
Run the local search and summarization app:

```bash
python app.py
```

Open `http://localhost:7860` in your browser. Type any natural language query to search reviews and get an AI-generated summary of what customers are saying.

**Example queries:**
- `coffee taste and flavor quality`
- `product arrived damaged packaging broken`
- `dog food ingredients and health benefits`
- `product good but packaging terrible`

---

## Dataset

**Source:** [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) (Kaggle)

| Stat | Value |
|---|---|
| Raw reviews | 568,454 |
| After cleaning | 567,070 |
| Date range | 2002 to 2012 |
| Unique users | 256,059 |
| Unique products | 74,258 |

Download `Reviews.csv` from Kaggle and place it in `data/raw/` before running the notebooks.

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_eda.ipynb` | Exploratory analysis, rating distribution, word frequency, temporal trends |
| `02_preprocessing.ipynb` | Cleaning, contraction expansion, lemmatization, dataset preparation |
| `03_topic_modeling.ipynb` | BERTopic clustering, topic visualization, sentiment by topic |
| `04_retrieval_summarization.ipynb` | FAISS index, semantic search, BART summarization, system evaluation |
| `05_classification.ipynb` | TF-IDF baseline, DistilBERT fine-tuning, error analysis, production recommendation |

---

## Key Findings

**Topic Modeling**
- Coffee and Hot Beverages dominates at 60%+ of all topic-assigned reviews
- Pet Food reviews have the highest helpfulness ratio (0.905), meaning buyers rely heavily on them before purchasing
- 9 clean topics discovered from 50,000 reviews with no manual labeling

**Semantic Search**
- Average cosine similarity above 0.70 across all test queries
- Sentiment filtering correctly isolates positive and negative review pools
- BART summarization condenses 15 reviews into a coherent paragraph

**Classification**

| Model | Accuracy | F1 Negative | ROC-AUC | Inference |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 88.08% | 0.7551 | 0.9432 | 0.38ms |
| DistilBERT (fine-tuned) | 91.00% | 0.7887 | 0.9518 | 75.42ms |

DistilBERT is 200x slower for a 3 point accuracy gain. Production recommendation: two-stage architecture where TF-IDF scores all reviews instantly and DistilBERT re-scores only low-confidence predictions (probability between 0.40 and 0.60).

---

## Tech Stack

| Category | Tools |
|---|---|
| Data processing | pandas, numpy, nltk |
| Visualization | matplotlib, seaborn, wordcloud |
| Topic modeling | BERTopic, UMAP, HDBSCAN |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector search | FAISS |
| Summarization | facebook/bart-large-cnn |
| Classification | scikit-learn, transformers, PyTorch |
| App | Gradio |

---

## Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/amazon-nlp-intelligence.git
cd amazon-nlp-intelligence

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
# Place Reviews.csv in data/raw/

# Run notebooks in order (01 through 05)
# Then launch the app
python app.py
```

---

## Project Structure

```
amazon-nlp-intelligence/
├── data/
│   ├── raw/              # Reviews.csv (download from Kaggle, not tracked)
│   └── processed/        # Generated files (not tracked)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_topic_modeling.ipynb
│   ├── 04_retrieval_summarization.ipynb
│   └── 05_classification.ipynb
├── outputs/
│   └── figures/          # All generated charts
├── app.py                # Gradio search and summarization demo
├── requirements.txt
└── README.md
```
