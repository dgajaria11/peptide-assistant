import requests
import json
import os
import time

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

SEARCH_TERMS = [
    "retatrutide",
    "semaglutide obesity weight loss",
    "tirzepatide weight loss",
    "glutathione weight loss",
    "BPC-157",
    "ipamorelin",
    "GLP-1 obesity",
    "AOD-9604"
]



def fetch_trials(search_term, max_results=20):
    """
    Searches ClinicalTrials.gov and returns a list of
    cleaned trial dictionaries for a given search term.
    """
    params = {
        "query.term": search_term,
        "pageSize": max_results,
        "format": "json"
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"  Error fetching '{search_term}': {e}")
        return []

    studies = data.get("studies", [])
    cleaned = []

    for study in studies:
        proto = study.get("protocolSection", {})

        # Each trial is broken into modules - we pull
        # the ones that are useful for our RAG pipeline
        id_module          = proto.get("identificationModule", {})
        status_module      = proto.get("statusModule", {})
        desc_module        = proto.get("descriptionModule", {})
        design_module      = proto.get("designModule", {})
        arms_module        = proto.get("armsInterventionsModule", {})
        outcomes_module    = proto.get("outcomesModule", {})
        eligibility_module = proto.get("eligibilityModule", {})

        cleaned.append({
            "nct_id":            id_module.get("nctId", ""),
            "title":             id_module.get("officialTitle", id_module.get("briefTitle", "")),
            "status":            status_module.get("overallStatus", ""),
            "phase":             design_module.get("phases", []),
            "summary":           desc_module.get("briefSummary", ""),
            "description":       desc_module.get("detailedDescription", ""),
            "interventions":     [i.get("name", "") for i in arms_module.get("interventions", [])],
            "primary_outcomes":  [o.get("measure", "") for o in outcomes_module.get("primaryOutcomes", [])],
            "eligibility":       eligibility_module.get("eligibilityCriteria", ""),
            "search_term":       search_term
        })

    return cleaned


def collect_all_trials(output_path="data/clinicaltrials_raw.json"):
    """
    Loops through all search terms, fetches trial data,
    deduplicates by NCT ID, and saves to JSON.
    """
    all_trials = []

    for term in SEARCH_TERMS:
        print(f"\nSearching trials for: {term}")
        trials = fetch_trials(term, max_results=20)
        print(f"  Found {len(trials)} trials")
        all_trials.extend(trials)
        time.sleep(0.5)

    # The same trial can appear under multiple search terms
    # e.g. a semaglutide trial might show up for both
    # "semaglutide obesity" and "GLP-1 obesity"
    # We use the NCT ID to keep only unique trials
    seen_ids = set()
    unique_trials = []
    for trial in all_trials:
        nct_id = trial["nct_id"]
        if nct_id and nct_id not in seen_ids:
            seen_ids.add(nct_id)
            unique_trials.append(trial)

    os.makedirs("data", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(unique_trials, f, indent=2)

    print(f"\nDone! Saved {len(unique_trials)} unique trials to {output_path}")
    return unique_trials


if __name__ == "__main__":
    collect_all_trials()