import torch
import json
import sys
from pathlib import Path
from tqdm import tqdm
from train_pot_dln_pointer import DLNPointerDecoder
import train_pot_clause as tpc
from train_pot_seq import load_rows

BASE_DIR = Path(__file__).resolve().parent

def score_gsm8k():
    print("--- Evaluating DLN Score on GSM8K Test Set ---")
    
    # 1. Load Data
    test_path = BASE_DIR / "data/gsm8k_test_lt.jsonl"
    vocab_path = BASE_DIR / "data/gsm8k_vocab.json"
    ckpt_path = BASE_DIR / "models/dln_gsm8k_best.pt"
    
    test_rows = load_rows(test_path)
    # Filter to evaluable rows
    test_rows = [r for r in test_rows if tpc.build_target_positions(r) is not None]
    print(f"Loaded {len(test_rows)} viable test examples.")
    
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
        
    model = DLNPointerDecoder(
        input_vocab=len(vocab), 
        max_positions=151,  # Max length from training 
        hidden_dim=128, 
        num_rules=16
    )
    
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    exact_match = 0
    clause_acc = 0.0
    total = len(test_rows)
    
    print("\nRunning Autoregressive Pointer Decoding...")
    with torch.no_grad():
        for row in tqdm(test_rows, desc="Scoring Test Set"):
            input_ids = torch.tensor([tpc.encode_input(row, vocab)], dtype=torch.long)
            
            # Predict
            dec_ids = torch.tensor([[0]], dtype=torch.long)
            out_clauses = []
            
            target_positions = tpc.build_target_positions(row)
            gold_tokens = [tpc.clause_text(prop) for prop in row["target_props"]]
            
            # Beam decode / greedy decode wrapper
            for step in range(len(gold_tokens) + 1):
                out = model(input_ids, dec_ids)
                logits = out[0] if isinstance(out, tuple) else out
                next_pos = logits[0, -1].argmax().item()
                
                if next_pos == input_ids.size(1):  # EOS
                    break
                    
                dec_ids = torch.cat([dec_ids, torch.tensor([[next_pos]])], dim=1)
                token_id = input_ids[0, next_pos].item()
                
                for k, v in vocab.items():
                    if v == token_id:
                        out_clauses.append(k)
                        break
            
            # Evaluate Accuracy
            if out_clauses == gold_tokens:
                exact_match += 1
                
            common = 0
            for a, b in zip(out_clauses, gold_tokens):
                if a == b: common += 1
            clause_acc += common / max(len(out_clauses), len(gold_tokens), 1)

    exact_pct = (exact_match / total) * 100
    clause_pct = (clause_acc / total) * 100
    
    print(f"\n--- Final GSM8K Scoring ---")
    print(f"Total Test Questions Evaluated: {total}")
    print(f"Exact Logical Form Match (System 1 Accuracy): {exact_pct:.2f}%")
    print(f"Clause-by-Clause Accuracy (Partial Credit): {clause_pct:.2f}%")
    print("---------------------------------------------")

if __name__ == "__main__":
    score_gsm8k()
