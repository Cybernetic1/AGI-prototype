import sys
import os
import json
from pathlib import Path
from tqdm import tqdm

# Absolute paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / 'gsm8k-tests'))
sys.path.append(str(BASE_DIR / 'gsm8k-tests' / 'lt_core'))

from spacy_logical_form import SpacyLogicalFormParser, canonicalize_form, parse_clause_line
from preprocess_gsm8k import load_jsonl

# The core boilerplate clauses we drop to keep sequences small for the prototype
CORE_DROP_PREDS = {"entity", "type", "tense", "question", "quantifier", "query_kind", "text"}

def filter_core_clauses(logical_form_text):
    """Filters out boilerplate clauses to make the sequence target smaller."""
    clauses = []
    for line in canonicalize_form(logical_form_text).splitlines():
        parsed = parse_clause_line(line)
        if parsed is None:
            continue
        pred, args = parsed
        if pred in CORE_DROP_PREDS:
            continue
        clauses.append({"pred": pred, "args": list(args)})
    return clauses

def convert_dataset(input_jsonl, output_jsonl, limit=None):
    print(f"Parsing {input_jsonl}...")
    try:
        examples = load_jsonl(input_jsonl)
    except FileNotFoundError:
        print(f"File not found: {input_jsonl}. Please run download_gsm8k.py first.")
        return
        
    if limit:
        examples = examples[:limit]
        
    parser = SpacyLogicalFormParser()
    out_rows = []
    
    for i, ex in enumerate(tqdm(examples, desc="Parsing sentences to Logical Forms")):
        text = ex["question"]
        
        # Parse the raw text into the Neo-Davidsonian logical form
        parsed_lf = parser.parse(text).render()
        
        # Filter down to the core clauses (Transfer, Agent, Patient, etc)
        core_clauses = filter_core_clauses(parsed_lf)
        
        row = {
            "id": ex.get("id", str(i)),
            "text": text,
            "answer": ex.get("answer", ""),
            "input_props": core_clauses,   # Input to DLN
            "target_props": core_clauses,  # For now, Auto-encoder objective (copying the structure)
            "family": "gsm8k"
        }
        out_rows.append(row)
        
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
    print(f"Wrote {len(out_rows)} examples to {output_jsonl}")

if __name__ == "__main__":
    # We will just parse a tiny subset first to verify the pipeline
    convert_dataset(
        "../data/gsm8k/main_test.jsonl", 
        "data/gsm8k_test_lt.jsonl",
        limit=20
    )
    convert_dataset(
        "../data/gsm8k/main_train.jsonl", 
        "data/gsm8k_train_lt.jsonl",
        limit=100
    )
