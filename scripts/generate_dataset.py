import json
import time
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Add scripts directory to path so we can import rag_pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import retrieve, build_prompt, generate_answer

QUESTIONS = [
    # Retatrutide
    "What are the side effects of taking reta?",
    "In what conditions should I not be taking reta?",
    "How much weight can I lose on retatrutide?",
    "Is retatrutide better than ozempic?",
    "Is reta FDA approved?",

    # Semaglutide / Tirzepatide
    "What is the difference between ozempic and mounjaro?",
    "Can I take semaglutide long term?",
    "What happens when you stop taking semaglutide?",
    "What are the risks of tirzepatide?",

    # BPC-157
    "Does BPC-157 help with weight loss?",
    "Is BPC-157 safe for humans?",
    "What does BPC-157 actually do?",
    "Can I take BPC-157 orally or does it need to be injected?",

    # TB-500
    "What are the long term implications of TB-500?",
    "Does TB-500 help with injury recovery?",
    "Is TB-500 legal?",

    # IGF-1
    "What are the risks of IGF-1 LR3?",
    "Does IGF-1 actually build muscle?",
    "Can IGF-1 cause cancer?",

    # GHK-Cu
    "What does GHK-Cu do for skin?",
    "Is GHK-Cu safe to use?",
    "Does copper peptide actually regrow hair?",

    # Ipamorelin
    "What does ipamorelin do?",
    "Is ipamorelin safer than HGH?",
    "Does ipamorelin actually work?",

    # Epitalon
    "What are the best anti-aging peptides?",
    "Does epitalon extend lifespan?",
    "Is epitalon backed by real science?",

    # Methylene Blue
    "What does methylene blue do to the brain?",
    "Is methylene blue safe?",
    "Can methylene blue improve memory?",

    # Tesamorelin
    "What is tesamorelin used for?",
    "Is tesamorelin FDA approved?",
    "How is tesamorelin different from HGH?",

    # General
    "What peptides are FDA approved for weight loss?",
    "Are peptides safe to self administer?",
    "What are the risks of buying peptides online?",
    "What peptides should I not stack together?",
    "How do peptides work in the body?",

    # Follow up and comparison questions
    "How does BPC-157 compare to NSAIDs for pain?",
    "What is the difference between ipamorelin and HGH?",
    "Can I stack BPC-157 and TB-500 together?",
    "What peptides are best for muscle recovery?",
    "How long does it take for semaglutide to work?",
    "What happens if I take too much retatrutide?",
    "Are there natural alternatives to GLP-1 peptides?",
    "What is the difference between peptides and steroids?",
    "Can women take the same peptides as men?",
    "What peptides help with inflammation?",
    "Is methylene blue a peptide?",
    "How do I know if a peptide source is trustworthy?",
    "What blood tests should I get before taking peptides?",
    "Can peptides interact with my medications?",
    "What is the glow stick peptide?",
    "How does tesamorelin differ from regular HGH therapy?",
    "What are the signs of peptide contamination?",
    "Can peptides help with autoimmune conditions?",
    "What does the research say about copper peptides for hair loss?",
    "Is epitalon worth trying for longevity?",

]


def generate_dataset(output_path="data/dataset.json"):
    dataset = []
    failed = []

    for i, question in enumerate(QUESTIONS):
        print(f"\n[{i+1}/{len(QUESTIONS)}] {question}")

        try:
            # Retrieve relevant chunks
            retrieved = retrieve(question, k=5)

            # Build the prompt
            prompt = build_prompt(question, retrieved)

            # Generate the answer
            answer = generate_answer(prompt)

            # Skip if answer is an error
            if answer.startswith("Error:"):
                print(f"  Failed: {answer}")
                failed.append(question)
                continue

            # Save the pair
            dataset.append({
                "instruction": question,
                "response": answer
            })

            print(f"  Generated answer ({len(answer)} chars)")

            # Wait between API calls to avoid rate limiting
            time.sleep(2)

        except Exception as e:
            print(f"  Exception: {e}")
            failed.append(question)
            continue

    # Save dataset
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDone!")
    print(f"  Successful: {len(dataset)}/{len(QUESTIONS)}")
    print(f"  Failed: {len(failed)}")
    if failed:
        print(f"  Failed questions: {failed}")
    print(f"  Saved to {output_path}")

    

def retry_failed(output_path="data/dataset.json"):
    """
    Loads existing dataset and retries any questions
    that didn't make it into the first run.
    """
    with open(output_path) as f:
        existing = json.load(f)

    completed_questions = {item["instruction"] for item in existing}
    remaining = [q for q in QUESTIONS if q not in completed_questions]

    print(f"Already completed: {len(completed_questions)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All questions completed!")
        return

    new_pairs = []
    for i, question in enumerate(remaining):
        print(f"\n[{i+1}/{len(remaining)}] {question}")
        try:
            retrieved = retrieve(question, k=5)
            prompt = build_prompt(question, retrieved)
            answer = generate_answer(prompt)

            if answer.startswith("Error:"):
                print(f"  Failed again: {answer}")
                continue

            new_pairs.append({
                "instruction": question,
                "response": answer
            })
            print(f"  Success ({len(answer)} chars)")
            time.sleep(2)

        except Exception as e:
            print(f"  Exception: {e}")
            continue

    # Merge with existing
    all_pairs = existing + new_pairs
    with open(output_path, "w") as f:
        json.dump(all_pairs, f, indent=2)

    print(f"\nDone! Total pairs now: {len(all_pairs)}")

if __name__ == "__main__":
    generate_dataset()
    print("\nRetrying any failed questions...")
    retry_failed()