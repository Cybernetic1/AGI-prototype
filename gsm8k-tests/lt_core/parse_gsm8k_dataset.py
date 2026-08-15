import json
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Ensure paths are absolute
BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE_DIR))
from preprocess_gsm8k import extract_propositions
from spacy_logical_form import SpacyLogicalFormParser

def parse_dataset(in_path, out_path, limit=None):
    parser = SpacyLogicalFormParser()
    rows = []
    
    with open(in_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    if limit:
        lines = lines[:limit]
        
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as out_f:
        for line in tqdm(lines, desc=f"Parsing {Path(in_path).name}"):
            data = json.loads(line)
            question = data["question"]
            
            # 1. Input Props: Raw linguistic hints/features (DLN input)
            raw_input_props = extract_propositions(question, source="question")
            input_props = [{"pred": str(p["pred"]), "args": [str(a) for a in p["args"]]} for p in raw_input_props]
            
            # 2. Target Props: Neo-Davidsonian logic (DLN target)
            lf = parser.parse(question)
            target_props = [{"pred": str(cl.predicate), "args": [str(a) for a in cl.args]} for cl in lf.clauses]
            
            row = {
                "text": question,
                "input_props": input_props,
                "target_props": target_props,
                "answer": data.get("answer", "")
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
    print(f"Wrote {len(lines)} parsed examples to {out_path}")

if __name__ == "__main__":
    train_in = BASE_DIR / "data" / "gsm8k" / "main_train.jsonl"
    test_in = BASE_DIR / "data" / "gsm8k" / "main_test.jsonl"
    train_out = BASE_DIR / "lt_core" / "data" / "gsm8k_train_lt.jsonl"
    test_out = BASE_DIR / "lt_core" / "data" / "gsm8k_test_lt.jsonl"
    
    parse_dataset(str(train_in), str(train_out), limit=None)
    parse_dataset(str(test_in), str(test_out), limit=None)
