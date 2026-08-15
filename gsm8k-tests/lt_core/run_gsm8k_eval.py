import torch
import sys
import os
from pathlib import Path
from tqdm import tqdm
import json

# Absolute paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / 'gsm8k-tests'))
sys.path.append(str(BASE_DIR / 'gsm8k-tests' / 'lt_core'))

from cognitive_loop import run_cognitive_loop

def run_evaluation():
    print("--- Running End-to-End GSM8K Evaluation ---")
    
    test_path = Path("data/gsm8k_test_lt.jsonl")
    with open(test_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # We will test a small slice for the demo
    test_slice = lines[:5]
    
    correct = 0
    
    for i, line in enumerate(test_slice):
        data = json.loads(line)
        question = data["text"]
        expected_answer = data["answer"]
        
        print(f"\n[{i+1}/{len(test_slice)}] Evaluating Problem:")
        
        # Run the cognitive loop!
        # The loop automatically prints its step-by-step logic
        final_wm = run_cognitive_loop(question, max_cycles=4)
        
        print(f"--- Question Finished ---")

if __name__ == "__main__":
    run_evaluation()
