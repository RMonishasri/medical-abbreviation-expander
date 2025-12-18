import sys
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt

# FIX: Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(ROOT)

import streamlit as st
from models.resolver import AbbreviationResolver

from models.embed_disambiguator import EmbedDisambiguator
from app.utils import find_abbreviations, highlight_expansions


st.set_page_config(page_title="🧠 Medical Abbreviation Expander + Evaluator", layout="wide")

# Initialize resolver
resolver = AbbreviationResolver()
use_embeddings = st.sidebar.checkbox("Enable contextual disambiguation (embeddings)", value=False)

disamb = None
if use_embeddings:
    with st.spinner("Loading embedding model..."):
        disamb = EmbedDisambiguator()

st.title(" Medical Abbreviation Expander + Evaluation Dashboard")

text_input = st.text_area("Paste clinical note text here", height=200)
uploaded = st.file_uploader("Or upload a .txt file", type=['txt'])
if uploaded and not text_input:
    text_input = uploaded.read().decode('utf-8')

if st.button("🔍 Expand Abbreviations"):
    if not text_input:
        st.warning("Please provide text or upload a file.")
    else:
        abbrs = find_abbreviations(text_input)
        expansions = {}

        for abbr in abbrs:
            candidates = resolver.lookup(abbr)
            if not candidates:
                expansions[abbr] = (f"[UNKNOWN: {abbr}]", 0.0)
                continue
            if len(candidates) == 1 or not use_embeddings:
                chosen, conf = resolver.resolve(abbr, context=text_input)
                expansions[abbr] = (chosen, conf)
            else:
                sentences = text_input.replace('\n', ' ').split('.')
                context_sentence = next((s for s in sentences if abbr in s), text_input)
                chosen, sim = disamb.choose_candidate(candidates, context_sentence)
                expansions[abbr] = (chosen, float(sim))

        # Display expansions
        st.subheader("Detected Abbreviations & Expansions")
        rows = [{"abbr": a, "expansion": e[0], "confidence": round(e[1], 3)} for a, e in expansions.items()]
        st.table(rows)

        # Highlighted annotated text
        st.markdown("### Annotated Text")
        annotated = highlight_expansions(text_input, expansions)
        st.markdown(annotated, unsafe_allow_html=True)

        # Download expanded text
        out_text = text_input
        for a, (exp, _) in expansions.items():
            out_text = out_text.replace(a, f"{a} ({exp})")
        st.download_button("💾 Download Expanded Text", data=out_text, file_name="expanded.txt")

        # === Analytics Logging ===
        try:
            os.makedirs("data", exist_ok=True)
            log_path = os.path.join("data", "analysis_data.csv")

            log_df = pd.DataFrame(rows)
            log_df["text_length"] = len(text_input)
            log_df["use_embeddings"] = use_embeddings
            log_df["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
                old = pd.read_csv(log_path)
                new = pd.concat([old, log_df], ignore_index=True)
            else:
                new = log_df

            new.to_csv(log_path, index=False, encoding="utf-8")
            st.success(f"✅ Analytics data saved successfully with {len(rows)} entries!")

        except Exception as e:
            st.error(f"⚠️ Could not save analytics data: {e}")

        # === Evaluation Metrics ===
        truth_path = os.path.join("data", "ground_truth.csv")
        if os.path.exists(truth_path):
            try:
                preds = pd.read_csv("data/analysis_data.csv")
                truth = pd.read_csv(truth_path)

                merged = pd.merge(preds, truth, on="abbr", how="inner")
                if not merged.empty:
                    merged["correct"] = merged.apply(
                        lambda x: 1 if x["expansion"].strip().lower() == x["true_expansion"].strip().lower() else 0, axis=1
                    )

                    precision = merged["correct"].sum() / len(merged)
                    accuracy = precision  # one prediction per abbr

                    st.markdown("## 📈 Model Evaluation Metrics")
                    col1, col2 = st.columns(2)
                    col1.metric("Precision", f"{precision:.2f}")
                    col2.metric("Accuracy", f"{accuracy:.2f}")

                    # --- Bar Chart (Fixed Version) ---
                    fig, ax = plt.subplots()
                    counts = merged["correct"].value_counts()
                    colors = ['#4CAF50' if i == 1 else '#F44336' for i in counts.index]
                    counts.plot(kind='bar', ax=ax, color=colors)
                    ax.set_title("Correct vs Incorrect Predictions")
                    labels = ["Correct" if i == 1 else "Incorrect" for i in counts.index]
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=0)
                    st.pyplot(fig)

                    # --- Save evaluation results ---
                    merged["precision"] = precision
                    merged["accuracy"] = accuracy
                    merged["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    merged.to_csv("data/evaluation_results.csv", index=False)
                    st.success("💾 Evaluation results saved to data/evaluation_results.csv")

                    # --- Trend Chart ---
                    evals = pd.read_csv("data/evaluation_results.csv")
                    if "timestamp" in evals.columns:
                        evals["timestamp"] = pd.to_datetime(evals["timestamp"])
                        avg_evals = evals.groupby("timestamp")[["precision", "accuracy"]].mean().reset_index()

                        fig2, ax2 = plt.subplots()
                        ax2.plot(avg_evals["timestamp"], avg_evals["precision"], marker='o', label="Precision")
                        ax2.plot(avg_evals["timestamp"], avg_evals["accuracy"], marker='s', label="Accuracy")
                        ax2.set_title("Precision & Accuracy Trend Over Time")
                        ax2.set_xlabel("Timestamp")
                        ax2.set_ylabel("Score")
                        ax2.legend()
                        plt.xticks(rotation=45)
                        st.pyplot(fig2)

                else:
                    st.info("ℹ️ No matching abbreviations found for evaluation.")
            except Exception as e:
                st.warning(f"⚠️ Could not compute evaluation metrics: {e}")
        else:
            st.info("⚠️ 'data/ground_truth.csv' not found. Please add it to enable evaluation.")
