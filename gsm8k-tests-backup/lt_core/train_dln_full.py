import torch
import torch.nn as nn
from pathlib import Path
from train_pot_seq import load_rows, split_rows
from train_pot_dln_pointer import DLNPointerDecoder
import train_pot_clause as tpc
import json
import sys

def train_gsm8k_overnight(epochs=40):
    print("--- Overnight Training: Logic Transformer on Full GSM8K Dataset ---", flush=True)
    
    train_path = Path("data/gsm8k_train_lt.jsonl")
    test_path = Path("data/gsm8k_test_lt.jsonl")
    
    print("Loading datasets...", flush=True)
    train_rows = load_rows(train_path)
    test_rows = load_rows(test_path)
    
    # Filter rows that can't be represented as a pointer sequence
    train_rows = [r for r in train_rows if tpc.build_target_positions(r) is not None]
    test_rows = [r for r in test_rows if tpc.build_target_positions(r) is not None]
    
    vocab = tpc.build_input_vocab(train_rows + test_rows)
    vocab_size = len(vocab)
    max_positions = max(len(r["input_props"]) for r in (train_rows + test_rows))
    print(f"Train size: {len(train_rows)}, Test size: {len(test_rows)}, Vocab size: {vocab_size}, Max Length: {max_positions}", flush=True)
    
    with open("data/gsm8k_vocab.json", "w") as f:
        json.dump(vocab, f)
        
    model = DLNPointerDecoder(
        input_vocab=vocab_size, 
        max_positions=max_positions, 
        hidden_dim=128, 
        num_rules=16
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    batch_size = 16
    train_batches = [train_rows[i:i + batch_size] for i in range(0, len(train_rows), batch_size)]
    test_batches = [test_rows[i:i + batch_size] for i in range(0, len(test_rows), batch_size)]
    
    print(f"\nStarting {epochs} Epochs of Training...", flush=True)
    best_loss = 9999.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_batches):
            optimizer.zero_grad()
            batch_loss = 0
            
            for row in batch:
                input_ids = torch.tensor([tpc.encode_input(row, vocab)], dtype=torch.long)
                target_positions = tpc.build_target_positions(row)
                decoder_input_ids = torch.tensor([tpc.encode_decoder_inputs(target_positions, max_positions + 1)], dtype=torch.long)
                
                logits = model(input_ids, decoder_input_ids)
                targets = torch.tensor([target_positions + [input_ids.size(1)]], dtype=torch.long)
                
                loss = nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                batch_loss += loss
                
            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += batch_loss.item()
            
            # Print batch progress every 50 steps so we can track it!
            if step % 50 == 0:
                print(f"  Epoch {epoch:02d} | Batch {step:03d}/{len(train_batches)} | Loss: {batch_loss.item():.4f}", flush=True)
            
        avg_train_loss = total_loss / len(train_batches)
        
        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_batches:
                batch_loss = 0
                for row in batch:
                    input_ids = torch.tensor([tpc.encode_input(row, vocab)], dtype=torch.long)
                    target_positions = tpc.build_target_positions(row)
                    decoder_input_ids = torch.tensor([tpc.encode_decoder_inputs(target_positions, max_positions + 1)], dtype=torch.long)
                    
                    logits = model(input_ids, decoder_input_ids)
                    targets = torch.tensor([target_positions + [input_ids.size(1)]], dtype=torch.long)
                    
                    loss = nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    batch_loss += loss
                val_loss += (batch_loss / len(batch)).item()
                
        avg_val_loss = val_loss / len(test_batches)
        
        print(f"=== Epoch {epoch:02d} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ===", flush=True)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = Path("models")
            save_path.mkdir(exist_ok=True)
            torch.save({'model_state': model.state_dict(), 'vocab': vocab}, save_path / "dln_gsm8k_best.pt")
            print("  [Saved new best model]", flush=True)
            
    print("\nTraining complete! Best model saved.", flush=True)

if __name__ == "__main__":
    train_gsm8k_overnight(40)
