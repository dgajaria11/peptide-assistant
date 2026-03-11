import streamlit as st
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Load from root directory when deployed on Hugging Face Spaces
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..') if os.path.exists(os.path.join(BASE_DIR, '..', 'faiss_index.bin')) else BASE_DIR

sys.path.append(os.path.join(BASE_DIR, '..', 'scripts'))

from rag_pipeline import retrieve, build_prompt, generate_answer

# Page configuration
st.set_page_config(
    page_title="Peptide Research Assistant",
    page_icon="🔬",
    layout="centered"
)

# Header
st.title("🔬 Peptide Research Assistant")
st.caption("Educational tool powered by published research. Not medical advice.")

st.warning(
    "⚠️ This tool is for educational purposes only. "
    "Always consult a healthcare provider before using any peptide or supplement. "
    "Many peptides discussed here are not FDA approved."
)

# Input
question = st.text_input(
    "Ask a question about peptides:",
    placeholder="e.g. What are the side effects of BPC-157?"
)

if st.button("Search Research", type="primary"):
    if not question.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Searching research database..."):
            retrieved = retrieve(question, k=5)

        with st.spinner("Generating answer..."):
            prompt = build_prompt(question, retrieved)
            answer = generate_answer(prompt)

        # Display answer
        st.subheader("Answer")
        st.markdown(answer)

        # Display sources
        st.subheader("Sources")
        for i, chunk in enumerate(retrieved):
            with st.expander(f"Source {i+1} — {chunk['source'].upper()} (score: {chunk['score']})"):
                st.write(chunk['text'])