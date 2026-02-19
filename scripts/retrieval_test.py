import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load everything we built
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("data/faiss_index.bin")

with open("data/chunks_indexed.json") as f:
    chunks = json.load(f)

def retrieve(query, k=3):
    """
    Takes a plain English question, embeds it,
    searches FAISS for the k most similar chunks,
    and returns them.
    """
    # Embed the query the same way we embedded the chunks
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype("float32")

    # Search the index — returns distances and indices
    distances, indices = index.search(query_vector, k)

    results = []
    for i, idx in enumerate(indices[0]):
        chunk = chunks[idx]
        results.append({
            "rank": i + 1,
            "score": round(float(distances[0][i]), 4),
            "source": chunk["source"],
            "text": chunk["text"][:400]  # first 400 chars
        })

    return results


# Test it with some sample questions
test_questions = [
    #"What are the side effects of taking reta?"   
    #"What is the mechanism of action of retatrutide?",
    #"What were the weight loss results in semaglutide trials?",
    #"What are the eligibility criteria for tirzepatide studies?",
    #"How does glutathione affect weight loss?"
]

for question in test_questions:
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")
    results = retrieve(question, k=3)
    for r in results:
        print(f"\nRank {r['rank']} | Source: {r['source']} | Score: {r['score']}")
        print(f"{r['text']}...")