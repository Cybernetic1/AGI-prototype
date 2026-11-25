"""Canonical learned-embedding LT models built on the vectorized logic core."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_logic_core_vectorized import VectorizedLogicNetwork


class LTSemanticContinuationModel(nn.Module):
    """Rank a true next sentence against an alternative using soft LT rules."""

    def __init__(
        self,
        vocab_size: int,
        *,
        symbol_dim: int = 64,
        num_rules: int = 16,
        max_sentence_tokens: int = 24,
        num_premises: int = 2,
        var_slots: int = 3,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.symbol_dim = int(symbol_dim)
        self.feature_dim = self.symbol_dim
        self.num_rules = int(num_rules)
        self.max_sentence_tokens = int(max_sentence_tokens)
        self.num_props = self.max_sentence_tokens * 2
        self.symbol_embed = nn.Embedding(self.vocab_size, self.symbol_dim, padding_idx=0)
        self.source_embed = nn.Embedding(2, self.symbol_dim)
        self.position_embed = nn.Embedding(self.max_sentence_tokens, self.symbol_dim)
        self.logic_core = VectorizedLogicNetwork(
            prop_length=self.symbol_dim,
            num_props=self.num_props,
            output_dim=self.symbol_dim,
            num_rules=self.num_rules,
            num_premises=num_premises,
            var_slots=var_slots,
        )
        self.pair_head = nn.Sequential(
            nn.Linear(self.symbol_dim * 2, self.symbol_dim),
            nn.ReLU(),
            nn.Linear(self.symbol_dim, 1),
        )

    def encode_side(
        self,
        token_ids: torch.Tensor,
        source_id: int,
    ) -> torch.Tensor:
        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        token_vectors = self.symbol_embed(token_ids)
        source = self.source_embed(
            torch.full_like(token_ids[:, :1], fill_value=source_id)
        ).expand(-1, token_ids.size(1), -1)
        position = self.position_embed(positions).unsqueeze(0).expand(token_ids.size(0), -1, -1)
        return token_vectors + source + position

    def score_candidate(
        self,
        context_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        context = self.encode_side(context_ids, 0)
        candidate = self.encode_side(candidate_ids, 1)
        working_memory = torch.cat([context, candidate], dim=1)
        logic_state = self.logic_core(working_memory)
        candidate_pool = candidate.mean(dim=1)
        return self.pair_head(torch.cat([logic_state, candidate_pool], dim=-1)).squeeze(-1)

    def forward(
        self,
        context_ids: torch.Tensor,
        candidate_a_ids: torch.Tensor,
        candidate_b_ids: torch.Tensor,
    ) -> torch.Tensor:
        candidates = torch.cat([candidate_a_ids, candidate_b_ids], dim=0)
        contexts = context_ids.repeat(2, 1)
        scores = self.score_candidate(contexts, candidates)
        batch_size = context_ids.size(0)
        return torch.stack(
            [scores[:batch_size], scores[batch_size:]],
            dim=1,
        )
