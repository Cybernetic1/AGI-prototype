from train_pot_clause import build_target_positions
from train_pot_seq import load_rows
from pathlib import Path

rows = load_rows(Path("data/gsm8k_train_lt.jsonl"))
print(f"Loaded {len(rows)} rows.")
row = rows[0]
print("Input Props:", row["input_props"])
print("Target Props:", row["target_props"])
print("Target Positions:", build_target_positions(row))
