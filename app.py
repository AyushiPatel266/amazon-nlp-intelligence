import gradio as gr
import pandas as pd
import numpy as np
import faiss
import torch
import os
from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer

# ─────────────────────────────────────────
# Load all models and data once at startup
# ─────────────────────────────────────────

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index("data/processed/faiss_index.bin")

print("Loading review dataset...")
df = pd.read_csv("data/processed/reviews_clean.csv")
df_retrieval = df.sample(n=100000, random_state=42).reset_index(drop=True)

print("Loading summarization model...")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

print("All models loaded. Starting app...")

# ─────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────

def semantic_search(query, top_k=15, sentiment_filter=None):
    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)

    search_k = top_k * 5 if sentiment_filter else top_k
    scores, indices = index.search(
        query_embedding.astype("float32"), search_k
    )

    results = df_retrieval.iloc[indices[0]].copy()
    results["similarity_score"] = scores[0]

    if sentiment_filter and sentiment_filter != "All":
        results = results[results["sentiment"] == sentiment_filter]

    return results.head(top_k).reset_index(drop=True)


def summarize_reviews(reviews_text_list, max_input_chars=3000,
                      max_length=180, min_length=60):
    combined = " | ".join(reviews_text_list)
    combined = combined[:max_input_chars]

    inputs = tokenizer(
        combined,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    summary_ids = bart_model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def search_and_summarize(query, sentiment_filter, num_results):
    if not query.strip():
        return "Please enter a search query.", "", pd.DataFrame()

    # Search
    results = semantic_search(
        query,
        top_k=int(num_results),
        sentiment_filter=sentiment_filter if sentiment_filter != "All" else None
    )

    if len(results) == 0:
        return "No results found for this query and filter combination.", "", pd.DataFrame()

    # Summarize
    summary = summarize_reviews(results["Text"].tolist())

    # Stats
    avg_sim = results["similarity_score"].mean()
    avg_rating = results["Score"].mean()
    stats = (
        f"Reviews retrieved: {len(results)}  |  "
        f"Avg similarity: {avg_sim:.4f}  |  "
        f"Avg star rating: {avg_rating:.2f}"
    )

    # Display table
    display_df = results[[
        "Score", "sentiment", "similarity_score", "Text"
    ]].copy()
    display_df.columns = ["Stars", "Sentiment", "Similarity", "Review Text"]
    display_df["Similarity"] = display_df["Similarity"].round(4)
    display_df["Review Text"] = display_df["Review Text"].str[:300] + "..."

    return summary, stats, display_df


# ─────────────────────────────────────────
# Gradio Interface
# ─────────────────────────────────────────

with gr.Blocks(title="Amazon Review Intelligence") as demo:

    gr.Markdown("""
    # Amazon Product Intelligence System
    ### Semantic Search and Summarization over 100,000 Customer Reviews
    
    Type any natural language query to search reviews semantically and get an 
    AI-generated summary of what customers are saying. Built with FAISS, 
    Sentence Transformers, and BART.
    """)

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="Search Query",
                placeholder="e.g. coffee taste and flavor quality",
                lines=2
            )
        with gr.Column(scale=1):
            sentiment_filter = gr.Dropdown(
                choices=["All", "Positive", "Negative"],
                value="All",
                label="Sentiment Filter"
            )
        with gr.Column(scale=1):
            num_results = gr.Slider(
                minimum=5,
                maximum=20,
                value=10,
                step=5,
                label="Number of Reviews"
            )

    search_btn = gr.Button("Search and Summarize", variant="primary")

    gr.Markdown("### AI-Generated Summary")
    summary_output = gr.Textbox(
        label="Summary",
        lines=4,
        interactive=False
    )

    stats_output = gr.Textbox(
        label="Search Statistics",
        lines=1,
        interactive=False
    )

    gr.Markdown("### Retrieved Reviews")
    results_table = gr.DataFrame(
        label="Most Relevant Reviews",
        wrap=True
    )

    gr.Markdown("""
    ---
    **Example queries to try:**
    - `coffee taste and flavor quality`
    - `product arrived damaged packaging broken`
    - `dog food ingredients and health benefits`
    - `tea aroma and brewing instructions`
    - `product expired bad smell returned`
    """)

    search_btn.click(
        fn=search_and_summarize,
        inputs=[query_input, sentiment_filter, num_results],
        outputs=[summary_output, stats_output, results_table]
    )

if __name__ == "__main__":
    demo.launch()