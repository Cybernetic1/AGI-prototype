from dataclasses import dataclass
from typing import Any, Optional, Dict

# ==========================================
# GWT Unified Fact Representation
# ==========================================
@dataclass(frozen=True)
class WMNode:
    concept: str
    attrs: frozenset
    activation: float = 1.0  # Continuous salience / confidence score

    @classmethod
    def create(cls, concept: str, activation: float = 1.0, **kwargs):
        # Filter out built-in 'activation' if passed in kwargs
        kwargs.pop("activation", None)
        return cls(concept, frozenset(sorted(kwargs.items())), activation)
    
    def to_dict(self):
        return dict(self.attrs)

    def __eq__(self, other):
        if not isinstance(other, WMNode):
            return False
        # Structural equality ignores activation (so we can find identical semantic facts)
        return self.concept == other.concept and self.attrs == other.attrs

    def __hash__(self):
        return hash((self.concept, self.attrs))

# ==========================================
# Working Memory (Global Workspace)
# ==========================================
class WorkingMemory:
    def __init__(self, default_ttl=3):
        # Dictionary mapping: WMNode -> remaining cycles (TTL)
        self.facts: Dict[WMNode, int] = {}
        self.default_ttl = default_ttl

    def update(self, new_facts):
        if not new_facts:
            return
            
        for fact in new_facts:
            # 1. Standard inventory state-updates
            if fact.concept == "Inventory":
                d = fact.to_dict()
                to_remove = [
                    f for f in self.facts 
                    if f.concept == "Inventory" 
                    and f.to_dict().get("owner") == d.get("owner") 
                    and f.to_dict().get("item") == d.get("item")
                ]
                for f in to_remove:
                    del self.facts[f]

            # 2. Competitive Selection (Salience/Activation check)
            # NOTE: If we migrate to Online Reinforcement Learning, do not delete/merge 
            # System 1's soft WMEs with System 2/3's hard WMEs, otherwise System 1 will 
            # lose gradient feedback (Credit Assignment failure). Instead, maintain origin 
            # labels or backpointers to reward System 1 for correct predictions.
            # If an structurally identical fact already exists:
            existing_match = next((f for f in self.facts if f == fact), None)
            if existing_match is not None:
                # Keep the one with HIGHER activation/confidence
                if fact.activation >= existing_match.activation:
                    # Remove the lower confidence node and insert the higher one
                    del self.facts[existing_match]
                    self.facts[fact] = self.default_ttl
                else:
                    # Keep existing higher-confidence node, but refresh its TTL
                    self.facts[existing_match] = self.default_ttl
            else:
                self.facts[fact] = self.default_ttl

    def decay(self):
        expired = []
        for fact in self.facts:
            self.facts[fact] -= 1
            if self.facts[fact] <= 0:
                expired.append(fact)
        for fact in expired:
            del self.facts[fact]
