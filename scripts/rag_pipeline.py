import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Load everything we built
model = SentenceTransformer("all-MiniLM-L6-v2")
# Works both locally and on Hugging Face Spaces
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, '..', 'data', 'faiss_index.bin')
CHUNKS_PATH = os.path.join(BASE_DIR, '..', 'data', 'chunks_indexed.json')

# Fall back to root directory for Hugging Face Spaces
if not os.path.exists(INDEX_PATH):
    INDEX_PATH = os.path.join(BASE_DIR, '..', 'faiss_index.bin')
    CHUNKS_PATH = os.path.join(BASE_DIR, '..', 'chunks_indexed.json')

index = faiss.read_index(INDEX_PATH)

with open(CHUNKS_PATH) as f:
    chunks = json.load(f)

print("RAG pipeline loaded successfully")



def retrieve(query, k=5):
    """
    Embeds the query and finds the k most relevant chunks.
    Returns the chunks with their metadata.
    """
    query_vector = np.array(model.encode([query])).astype("float32")
    distances, indices = index.search(query_vector, k)

    results = []
    for i, idx in enumerate(indices[0]):
        chunk = chunks[idx]
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "score": round(float(distances[0][i]), 4)
        })
    return results



def build_prompt(question, retrieved_chunks):
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(f"[Source {i+1} - {chunk['source']}]\n{chunk['text']}")
    
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an educational assistant that helps people understand peptides and their effects based on published research.

Your role is to:
- Explain what the research actually says, including both benefits AND risks
- Clearly note when a peptide is NOT FDA approved
- Use plain language that someone without a science background can understand
- Always remind users this is for educational purposes only, not medical advice
- Be honest when research is limited or inconclusive
- Never say "you've shared" or "the research you provided" — the research comes from a database, not from the user

Use ONLY the provided research context to answer. Do not make up information.
If the context does not contain enough information to answer, say so clearly.
Refer to sources as "published research" or "studies" not as something the user shared.

RESEARCH CONTEXT:
{context}

USER QUESTION: {question}

EDUCATIONAL ANSWER:"""

    return prompt




def generate_answer(prompt):
    token = os.getenv("HF_TOKEN")
    client = InferenceClient(token=token)
    
    
    response = client.chat_completion(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=750,
        temperature=0.3
    )
    
    return response.choices[0].message.content


def ask(question, k=5):
    """
    Full RAG pipeline — retrieves relevant chunks,
    builds a prompt, and generates a real answer.
    """
    print(f"\nQuestion: {question}")
    print("Retrieving relevant research...")
    
    retrieved = retrieve(question, k=k)
    prompt = build_prompt(question, retrieved)
    
    print("Generating answer...")
    answer = generate_answer(prompt)
    
    print("\n--- ANSWER ---")
    print(answer)
    print("\n--- SOURCES ---")
    for i, r in enumerate(retrieved):
        print(f"  {i+1}. [{r['source']}] score: {r['score']}")
    
    return answer, retrieved


if __name__ == "__main__":
    test_questions = [
        "What are the risks of taking BPC-157?",
        "How does retatrutide cause weight loss?",
        "Is epitalon backed by real science?"
    ]
    
    for question in test_questions:
        ask(question)
        print("\n" + "="*60)


if __name__ == "__main__":
    # Test the full pipeline
    test_questions = [
        "What are the risks of taking BPC-157?",
        "How does retatrutide cause weight loss?",
        "Is epitalon backed by real science?"
    ]
    
    for question in test_questions:
        prompt, retrieved = ask(question)
        print("\n--- PROMPT PREVIEW (first 500 chars) ---")
        print(prompt[:500])
        print("...")