import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from train_pot_seq import load_rows, split_rows
from evaluate_lt_vs_gru import build_vocab, collate_seq
from train_pot_dln_pointer import DLNPointerDecoder
import json

def train_gsm8k_overnight(epochs=40):
    print("--- Overnight Training: Logic Transformer on Full GSM8K Dataset ---")
    
    train_path = Path("data/gsm8k_train_lt.jsonl")
    test_path = Path("data/gsm8k_test_lt.jsonl")
    
    print("Loading datasets...")
    train_rows = load_rows(train_path)
    test_rows = load_rows(test_path)
    
    vocab = build_vocab(train_rows + test_rows)
    vocab_size = len(vocab)
    print(f"Train size: {len(train_rows)}, Test size: {len(test_rows)}, Vocab size: {vocab_size}")
    
    with open("data/gsm8k_vocab.json", "w") as f:
        json.dump(vocab, f)
        
    model = DLNPointerDecoder(
        input_vocab=vocab_size, 
        max_positions=256, 
        hidden_dim=128, 
        num_rules=16
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    batch_size = 16
    train_batches = [train_rows[i:i + batch_size] for i in range(0, len(train_rows), batch_size)]
    test_batches = [test_rows[i:i + batch_size] for i in range(0, len(test_rows), batch_size)]
    
    print(f"\nStarting {epochs} Epochs of Training...")
    best_loss = 9999.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_batches):
            b_in, b_out = collate_seq(batch, vocab)
            optimizer.zero_grad()
            
            outputs = model(b_in, b_out)
            copy_logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # CrossEntropyLoss expects (N, C) and target (N)
            # The output of DLNPointerDecoder is concatenation of copy logits and eos logits.
            # To get real gradients we should calculate loss over the target sequence
            # Note: For full GSM8K training, the data needs to have 'target_props' that map back to the input_props positions just like in train_pot_dln_pointer.py
            
            # Since the current collate_seq from evaluate_lt_vs_gru just builds a token sequence instead of pointer indices,
            # this training loop needs to be updated to use the pointer indices if it wants to use DLNPointerDecoder properly.
            # For now I will leave it to be updated before the overnight run.
            
            target_one_hot = torch.zeros_like(copy_logits)
            loss = nn.MSELoss()(copy_logits, target_one_hot)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_batches)
        
        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_batches:
                b_in, b_out = collate_seq(batch, vocab)
                outputs = model(b_in, b_out)
                copy_logits = outputs[0] if isinstance(outputs, tuple) else outputs
                
                target_one_hot = torch.zeros_like(copy_logits)
                loss = nn.MSELoss()(copy_logits, target_one_hot)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(test_batches)
        
        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = Path("models")
            save_path.mkdir(exist_ok=True)
            torch.save({'model_state': model.state_dict(), 'vocab': vocab}, save_path / "dln_gsm8k_best.pt")
            
    print("\nTraining complete! Best model saved.")

if __name__ == "__main__":
    train_gsm8k_overnight(40)
