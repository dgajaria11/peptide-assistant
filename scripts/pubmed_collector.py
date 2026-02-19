import json
import requests
import time
import os

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

SEARCH_QUERIES = [
    # GLP-1 / Weight loss peptides
    "retatrutide weight loss clinical trial",
    "retatrutide safety side effects",
    "semaglutide obesity clinical trial",
    "tirzepatide body composition",
    "tesamorelin growth hormone fat loss",
    "tesamorelin safety side effects",
    "AOD-9604 fat metabolism",

    # Growth hormone related
    "ipamorelin growth hormone efficacy",
    "ipamorelin safety clinical evidence",
    "growth hormone secretagogue body composition",
    "growth hormone risks side effects adults",
    "insulin-like growth factor muscle growth",
    "IGF-1 risks side effects insulin resistance",

    # Recovery and healing peptides
    "BPC-157 peptide therapy healing",
    "BPC-157 safety side effects human",
    "thymosin beta-4 tissue repair recovery",
    "thymosin beta-4 safety adverse effects",

    # Skin, hair, longevity
    "copper peptide GHK skin regeneration",
    "copper peptide wound healing",
    "epithalamin pineal peptide aging",
    "epitalon aging telomerase",

    # Antioxidants and other
    "glutathione weight loss oxidative stress",
    "methylene blue cognitive effects",
    "methylene blue safety risks",

    # Broad educational queries
    "peptide therapy risks unapproved",
    "GLP-1 agonist weight loss mechanism",
    "unapproved peptides human safety risks",
]

def search_pubmed(query, max_results=50):
    url = f"{PUBMED_BASE}esearch.fcgi"
    
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Add this to see exactly what's coming back
    if "esearchresult" not in data:
        print(f"  Unexpected response for '{query}': {data}")
        return []
    
    ids = data["esearchresult"]["idlist"]
    print(f"  Found {len(ids)} articles for: '{query}'")
    return ids


def fetch_abstracts(pmids):
    """
    Step 2: Given a list of article IDs, fetch the actual abstract text.
    This is the content that will go into our RAG knowledge base.
    """
    if not pmids:
        return ""
    
    url = f"{PUBMED_BASE}efetch.fcgi"
    
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),  # pass all IDs at once, comma separated
        "rettype": "abstract",  # we want abstracts, not full text
        "retmode": "text"       # plain text format
    }
    
    response = requests.get(url, params=params)
    return response.text



def collect_all_data(output_path="data/pubmed_raw.json"):
    """
    Loops through all our queries, searches PubMed, fetches abstracts,
    and saves everything to a JSON file.
    """
    all_documents = []
    
    for query in SEARCH_QUERIES:
        print(f"\nSearching for: {query}")
        
        # Step 1: get IDs
        pmids = search_pubmed(query, max_results=50)
        
        if not pmids:
            print("  No results found, skipping.")
            continue
        
        # Step 2: fetch the actual text
        abstract_text = fetch_abstracts(pmids)
        
        # Store both the query and the content together
        all_documents.append({
            "query": query,
            "pmids": pmids,
            "content": abstract_text
        })
        
        # IMPORTANT: PubMed asks that you wait between requests
        # so you don't overload their servers
        print(f"  Fetched abstracts. Waiting 1 second...")
        time.sleep(1)
    
    # Save everything to a JSON file
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_documents, f, indent=2)
    
    print(f"\nDone! Saved {len(all_documents)} query batches to {output_path}")
    return all_documents


# This makes the script runnable directly
if __name__ == "__main__":
    collect_all_data()