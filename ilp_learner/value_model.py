from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


class LinearValueModel:
    """Lightweight linear scorer for state-action features.

    This is intentionally simple for first integration: it provides a stable,
    explainable value prior for value-guided clause search.
    """

    def __init__(self, weights: Dict[str, float] | None = None, bias: float = 0.0):
        self.weights = dict(weights or {})
        self.bias = float(bias)

    def score(self, features: Dict[str, float]) -> float:
        total = self.bias
        for key, val in features.items():
            total += self.weights.get(key, 0.0) * float(val)
        return float(total)

    def update(self, features: Dict[str, float], target: float, lr: float = 0.01) -> None:
        pred = self.score(features)
        err = float(target) - pred
        self.bias += lr * err
        for key, val in features.items():
            self.weights[key] = self.weights.get(key, 0.0) + lr * err * float(val)

    def to_dict(self) -> Dict[str, object]:
        return {"bias": self.bias, "weights": self.weights}

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "LinearValueModel":
        return cls(weights=dict(payload.get("weights") or {}), bias=float(payload.get("bias", 0.0)))

    @classmethod
    def load(cls, path: str | Path) -> "LinearValueModel":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

