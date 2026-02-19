import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

def clean_text(text):
    """
    Removes PubMed citation noise from chunks.
    Things like author lists, DOI lines, PMID lines,
    and journal headers that don't contain useful content.
    """
    # Remove DOI lines
    text = re.sub(r'DOI:.*?\n', '', text)
    
    # Remove PMID lines
    text = re.sub(r'PMID:.*?\n', '', text)
    
    # Remove PMCID lines
    text = re.sub(r'PMCID:.*?\n', '', text)
    
    # Remove author information lines
    text = re.sub(r'Author information:.*?\n', '', text)
    
    # Remove conflict of interest statements
    text = re.sub(r'Conflict of interest.*?(\n\n|\Z)', '', text, flags=re.DOTALL)
    
    # Remove copyright lines
    text = re.sub(r'Copyright ©.*?\n', '', text)
    
    # Remove lines that are just numbers and punctuation (citation numbers)
    text = re.sub(r'^\d+\.\s*\n', '', text, flags=re.MULTILINE)
    
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    return text

# This is the splitter we'll use to break text into chunks.
# chunk_size: maximum size of each chunk in characters
# chunk_overlap: how many characters overlap between chunks
# Overlap is important — it prevents a key idea from being
# cut in half at a chunk boundary and losing its context
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)

def process_pubmed(input_path="data/pubmed_raw.json"):
    """
    Takes the raw PubMed data and splits each query's
    content blob into individual chunks with metadata.
    """
    with open(input_path) as f:
        data = json.load(f)

    all_chunks = []

    for entry in data:
        query = entry["query"]
        content = entry["content"]

        # Skip if content is empty
        if not content.strip():
            continue

        #clean
        content = clean_text(content)


        # Split the big text blob into chunks
        chunks = splitter.split_text(content)

        for chunk in chunks:
            # Skip very short chunks — they're usually
            # just headers or formatting artifacts
            if len(chunk.strip()) < 100:
                continue

            all_chunks.append({
                "text": chunk.strip(),
                "source": "pubmed",
                "query": query
            })

    print(f"PubMed: created {len(all_chunks)} chunks from {len(data)} queries")
    return all_chunks

def process_clinicaltrials(input_path="data/clinicaltrials_raw.json"):
    """
    Takes the raw ClinicalTrials data and converts each trial
    into one or more chunks. Unlike PubMed, this data is already
    structured so we build a readable text block per trial first,
    then chunk it.
    """
    with open(input_path) as f:
        data = json.load(f)

    all_chunks = []

    for trial in data:
        # Build a readable text block from the structured fields
        # This is important — we're converting JSON structure
        # into natural language so it embeds properly
        parts = []

        if trial.get("title"):
            parts.append(f"Trial Title: {trial['title']}")

        if trial.get("status"):
            parts.append(f"Status: {trial['status']}")

        if trial.get("phase"):
            parts.append(f"Phase: {', '.join(trial['phase'])}")

        if trial.get("summary"):
            parts.append(f"Summary: {trial['summary']}")

        if trial.get("description"):
            parts.append(f"Description: {trial['description']}")

        if trial.get("interventions"):
            parts.append(f"Interventions: {', '.join(trial['interventions'])}")

        if trial.get("primary_outcomes"):
            parts.append(f"Primary Outcomes: {', '.join(trial['primary_outcomes'])}")

        if trial.get("eligibility"):
            parts.append(f"Eligibility: {trial['eligibility']}")

        # Join all parts into one text block
        full_text = "\n\n".join(parts)

        if not full_text.strip():
            continue

        # Chunk it — longer trials may produce multiple chunks
        chunks = splitter.split_text(full_text)

        for chunk in chunks:
            if len(chunk.strip()) < 100:
                continue

            all_chunks.append({
                "text": chunk.strip(),
                "source": "clinicaltrials",
                "nct_id": trial.get("nct_id", ""),
                "title": trial.get("title", "")
            })

    print(f"ClinicalTrials: created {len(all_chunks)} chunks from {len(data)} trials")
    return all_chunks


def main():
    print("Processing PubMed data...")
    pubmed_chunks = process_pubmed()

    print("\nProcessing ClinicalTrials data...")
    trial_chunks = process_clinicaltrials()

    # Combine everything into one list
    all_chunks = pubmed_chunks + trial_chunks

    # Save the chunks to disk
    os.makedirs("data", exist_ok=True)
    with open("data/chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nDone! Total chunks saved: {len(all_chunks)}")
    print(f"  PubMed chunks:        {len(pubmed_chunks)}")
    print(f"  ClinicalTrials chunks: {len(trial_chunks)}")
    print(f"\nSaved to data/chunks.json")


if __name__ == "__main__":
    main()