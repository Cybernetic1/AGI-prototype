import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LT_DIR = REPO_ROOT / "logic_transformer"
if str(LT_DIR) not in sys.path:
    sys.path.insert(0, str(LT_DIR))

from model import LTSemanticContinuationModel
from preprocess_gsm8k import build_continuation_pairs, extract_propositions


class TinyStoriesSemanticTrainingTest(unittest.TestCase):
    def test_pairs_do_not_mix_stories_between_splits(self):
        stories = [
            f"Story {i} starts here. Then child {i} walks home. Finally child {i} sleeps."
            for i in range(10)
        ]
        pairs = build_continuation_pairs(stories, validation_ratio=0.3, seed=7)
        train_stories = {row["story_id"] for row in pairs if row["split"] == "train"}
        validation_stories = {row["story_id"] for row in pairs if row["split"] == "validation"}
        self.assertTrue(train_stories)
        self.assertTrue(validation_stories)
        self.assertFalse(train_stories & validation_stories)
        self.assertTrue(all(row["label"] in {0, 1} for row in pairs))

    def test_continuation_model_emits_two_candidate_logits(self):
        model = LTSemanticContinuationModel(feature_dim=16)
        context = extract_propositions("The rabbit entered the garden.", source="question")
        candidate_a = extract_propositions("It looked at the flowers.", source="question")
        candidate_b = extract_propositions("The moon floated over the sea.", source="question")
        logits = model(context, candidate_a, candidate_b)
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
