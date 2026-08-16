from train_pot_seq import load_rows
import train_pot_clause as tpc
import json
from pathlib import Path

BASE_DIR = Path("/home/yky/misc-programs/AGI-prototype")
test_rows = load_rows(BASE_DIR / "gsm8k-tests/lt_core/data/gsm8k_test_lt.jsonl")
test_rows = [r for r in test_rows if tpc.build_target_positions(r) is not None]

row = test_rows[0]
gold_tokens = [tpc.clause_text(prop) for prop in row["target_props"]]
print("Gold Tokens Length:", len(gold_tokens))
print("First 5 Gold Tokens:", gold_tokens[:5])

input_tokens = [tpc.clause_text(prop) for prop in row["input_props"]]
print("\nInput Tokens Length:", len(input_tokens))
print("First 5 Input Tokens:", input_tokens[:5])

targets = tpc.build_target_positions(row)
print("\nTarget Indices:", targets)
