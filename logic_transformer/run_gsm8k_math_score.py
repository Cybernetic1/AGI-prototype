import torch
import json
import sys
import re
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / 'gsm8k-tests'))
sys.path.append(str(BASE_DIR / 'gsm8k-tests' / 'lt_core'))

from train_pot_dln_pointer import DLNPointerDecoder
import train_pot_clause as tpc
from train_pot_seq import load_rows
from cognitive_loop import run_cognitive_loop, _dln_model

def extract_gold_answer(answer_text):
    """Extracts the final numeric answer from GSM8K's '#### 123' format."""
    match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', str(answer_text))
    if match:
        return float(match.group(1))
    # Fallback to the last number found
    numbers = re.findall(r'-?\d+(?:\.\d+)?', str(answer_text))
    if numbers:
        return float(numbers[-1])
    return None

def score_math_accuracy():
    print("--- Evaluating End-to-End Math Accuracy on GSM8K Test Set ---")
    
    test_path = BASE_DIR / "gsm8k-tests/lt_core/data/gsm8k_test_lt.jsonl"
    test_rows = load_rows(test_path)
    
    # Take a representative slice to avoid running Experta 1300 times in python loop sequentially
    test_rows = test_rows[:100] 
    
    print(f"Loaded {len(test_rows)} test examples.")
    
    correct = 0
    total = len(test_rows)
    
    for i, row in enumerate(tqdm(test_rows, desc="Solving Math Problems")):
        question = row["text"]
        gold_answer_val = extract_gold_answer(row["answer"])
        
        # Run the full cognitive loop (System 1 DLN -> System 2 Experta)
        final_wm = run_cognitive_loop(question, max_cycles=4, quiet=True)
        
        # We look for a Belief(answer=X) or the final state of an Inventory/Quantity
        predicted_answer = None
        for fact in final_wm.facts:
            # We look for whatever Experta output as the "final answer"
            if fact.concept == "Belief" and fact.to_dict().get("label") == "final_answer":
                predicted_answer = float(fact.to_dict().get("value", 0))
                break
        
        # If Experta didn't produce a 'final_answer' belief, we might need to dig into Inventories
        # But ideally System 2 axioms should conclude with a final answer.
        
        if predicted_answer is not None and gold_answer_val is not None:
            if abs(predicted_answer - gold_answer_val) < 1e-5:
                correct += 1
                
    accuracy = (correct / total) * 100
    print(f"\n--- Final GSM8K Math Accuracy ---")
    print(f"Total Test Questions Evaluated: {total}")
    print(f"Correct Answers (System 1 + System 2): {accuracy:.2f}%")
    print("---------------------------------------------")

if __name__ == "__main__":
    score_math_accuracy()
