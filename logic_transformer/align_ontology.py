import sys
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

# Add paths to import ontology and tpc
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR / 'forward-engine'))
sys.path.append(str(BASE_DIR / 'gsm8k-tests' / 'lt_core'))
sys.path.append(str(BASE_DIR / 'logic-transformer'))

from phase3_synthetic_test import ontology_classify
import train_pot_clause as tpc
from lorentz_math import project_to_hyperboloid, lorentz_distance

def get_predicate(clause_str):
    if "(" not in clause_str:
        return None
    return clause_str.split("(")[0].strip()

def align_embeddings(data_path, output_path, hidden_dim=128, epochs=500, lr=0.01, margin=2.0):
    rows = tpc.load_rows(Path(data_path))
    vocab = tpc.build_input_vocab(rows)
    print(f"Vocab size: {len(vocab)}")

    # Classify each vocab item
    vocab_classes = {}
    valid_verbs = 0
    for token, idx in vocab.items():
        if token.startswith("<"):
            continue
        pred = get_predicate(token)
        if pred:
            cls = ontology_classify(pred)
            if cls != "neutral":
                vocab_classes[idx] = cls
                valid_verbs += 1
    
    print(f"Found {valid_verbs} tokens with active ontology classes.")

    # Initialize embeddings
    embeddings = torch.nn.Embedding(len(vocab), hidden_dim)
    torch.nn.init.normal_(embeddings.weight, std=0.1)
    
    optimizer = optim.Adam(embeddings.parameters(), lr=lr)
    
    # Get all indices with classes
    indices = list(vocab_classes.keys())
    classes = [vocab_classes[i] for i in indices]
    
    # Create mask for positive and negative pairs
    n = len(indices)
    pos_mask = torch.zeros(n, n, dtype=torch.bool)
    neg_mask = torch.zeros(n, n, dtype=torch.bool)
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if classes[i] == classes[j]:
                pos_mask[i, j] = True
            else:
                neg_mask[i, j] = True
                
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        hyper_embs = project_to_hyperboloid(embeddings.weight)
        
        # Get only the relevant embeddings
        valid_embs = hyper_embs[indices_tensor] # (N, L)
        
        # Compute all pairwise Lorentz distances
        # valid_embs.unsqueeze(1): (N, 1, L)
        # valid_embs.unsqueeze(0): (1, N, L)
        distances = lorentz_distance(valid_embs.unsqueeze(1), valid_embs.unsqueeze(0)) # (N, N)
        
        pos_loss = distances[pos_mask].sum()
        neg_loss = F.relu(margin - distances[neg_mask]).sum()
        
        num_pos = pos_mask.sum().item()
        num_neg = neg_mask.sum().item()
        
        loss = (pos_loss + neg_loss) / (num_pos + num_neg)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Contrastive Loss: {loss.item():.4f}")
            
    # Save the aligned raw weights (they will be projected on-the-fly in DLN)
    torch.save(embeddings.state_dict(), output_path)
    print(f"Saved aligned embeddings to {output_path}")

if __name__ == '__main__':
    align_embeddings(
        data_path='/tmp/pot_lt_pairs_clean_balanced2.jsonl',
        output_path='/tmp/aligned_hyperbolic_embeddings.pt'
    )
