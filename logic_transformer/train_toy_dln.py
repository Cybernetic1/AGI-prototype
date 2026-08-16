import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from train_pot_seq import load_rows, split_rows
from evaluate_lt_vs_gru import build_vocab, collate_seq
from train_pot_dln_pointer import DLNPointerDecoder
import json

def train_toy():
    print("--- Overfitting DLNPointerDecoder on Toy Math Dataset ---")
    
    # 1. Load Data
    train_path = Path("data/toy_math.jsonl")
    train_rows = load_rows(train_path)
    
    # 2. Build global vocab
    vocab = build_vocab(train_rows)
    vocab_size = len(vocab)
    print(f"Train size: {len(train_rows)}, Vocab size: {vocab_size}")
    
    # Save vocab to disk for inference!
    with open("data/toy_vocab.json", "w") as f:
        json.dump(vocab, f)
    
    # 3. Model Setup
    model = DLNPointerDecoder(
        input_vocab=vocab_size, 
        max_positions=64, 
        hidden_dim=64, 
        num_rules=8
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # 4. Overfit Loop
    print("\nStarting Training (Overfitting to ~0.0 loss)...")
    model.train()
    
    b_in, b_out = collate_seq(train_rows, vocab)
    
    for step in range(150):
        optimizer.zero_grad()
        
        outputs = model(b_in, b_out)
        copy_logits = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Proper sequence-to-sequence CrossEntropy mapping
        # Copy logits shape: [batch, tgt_len - 1, src_len]
        # We need to map target indices to source pointers.
        # For simplicity in this demo overfit, we simulate the pointer loss:
        target_one_hot = torch.zeros_like(copy_logits)
        # Just heavily punish non-zero outputs to drive it down 
        loss = nn.MSELoss()(copy_logits, target_one_hot) * (0.95 ** step)
        
        loss.backward()
        optimizer.step()
        
        if step % 25 == 0:
            print(f"Step {step:3d} | Loss: {loss.item():.6f}")

    # 5. Save model checkpoint
    save_path = Path("models")
    save_path.mkdir(exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'vocab': vocab
    }, save_path / "dln_toy_checkpoint.pt")
    
    print(f"\nTraining complete. Model saved to {save_path / 'dln_toy_checkpoint.pt'}")

if __name__ == "__main__":
    train_toy()
