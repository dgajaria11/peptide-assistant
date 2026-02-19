import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# This is the embedding model we're using.
# It converts text into a 384-dimensional vector.
# It's small, fast, and works great for semantic search.
model = SentenceTransformer("all-MiniLM-L6-v2")

def build_index(chunks_path="data/chunks.json"):
    """
    Loads all chunks, embeds them using sentence-transformers,
    and saves a FAISS index to disk so we can search it later.
    """
    print("Loading chunks...")
    with open(chunks_path) as f:
        chunks = json.load(f)

    # Extract just the text from each chunk
    # This is what gets embedded
    texts = [chunk["text"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunks — this may take a few minutes...")

    # Convert all texts to vectors
    # show_progress_bar gives you a live progress indicator
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64
    )

    # Convert to float32 — FAISS requires this specific type
    embeddings = np.array(embeddings).astype("float32")

    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"  {embeddings.shape[0]} chunks embedded")
    print(f"  {embeddings.shape[1]} dimensions per vector")

    # Build the FAISS index
    # IndexFlatL2 uses L2 (euclidean) distance to find
    # the most similar vectors to a query
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"\nFAISS index built with {index.ntotal} vectors")

    # Save everything to disk
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, "data/faiss_index.bin")

    # Save the chunks separately so we can look up
    # the original text after finding a match
    with open("data/chunks_indexed.json", "w") as f:
        json.dump(chunks, f, indent=2)

    print("Saved faiss_index.bin and chunks_indexed.json")
    return index, chunks


if __name__ == "__main__":
    build_index()