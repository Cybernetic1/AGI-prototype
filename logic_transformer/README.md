# Logic Transformer (LT)

This directory contains the maintained neural-symbolic LT implementation.

## Canonical models

- `neural_logic_core_vectorized.py`: primary vectorized differentiable logic core.
- `train_pot_dln_pointer.py`: DLN pointer decoder used by the cognitive-loop/LT
  proposition-copy path.
- `train.py`: resumable batched semantic-continuation trainer for real TinyStories
  sentence pairs.
- `model.py`: learned-symbol embedding model for semantic continuation, built on
  the vectorized logic core.

The semantic-continuation model uses trainable symbol, source, and position
embeddings. Predicate/word identity is not encoded as a normalized arbitrary hash
number.

## TinyStories semantic-continuation training

Prepare data from an existing separator-delimited 10,000-story file and train:

```bash
/home/yky/misc-programs/.venv/bin/python logic_transformer/train.py \
  --tinystories-source ../AGI-experiments/data/tinystories/stories_10000.txt \
  --prepared-output runtime/tinystories/semantic_continuation_10k.jsonl \
  --max-stories 10000 \
  --epochs 20 \
  --batch-size 32 \
  --num-rules 16 \
  --symbol-dim 64 \
  --checkpoint checkpoints/lt_semantic_continuation.pt
```

Resume with the same prepared data and `--resume`.

## Existing LT pointer-decoder path

`train_pot_dln_pointer.py` is the established proposition-copy trainer used by the
integrated cognitive-loop architecture. The older compact GSM arithmetic baseline
is retained only under `gsm8k-tests-backup/` and is explicitly not canonical LT.

## Supporting modules

- `spacy_logical_form.py`: Neo-Davidsonian parsing
- `semantic_ar.py`: semantic autoregressive scoring
- `lorentz_math.py`: hyperbolic matching geometry
- `local_proposal_ranker.py`: optional proposal ranking
- `train_pot_clause.py`: proposition/clause encoding utilities

Demo, download, and sweep scripts are support tooling rather than the core model.
