from dataclasses import dataclass

# ==========================================
# GWT Unified Fact Representation
# ==========================================
@dataclass(frozen=True)
class WMNode:
    concept: str
    attrs: frozenset

    @classmethod
    def create(cls, concept: str, **kwargs):
        return cls(concept, frozenset(sorted(kwargs.items())))
    
    def to_dict(self):
        return dict(self.attrs)

# ==========================================
# Working Memory (Global Workspace)
# ==========================================
class WorkingMemory:
    def __init__(self, default_ttl=3):
        self.facts = {}
        self.default_ttl = default_ttl

    def update(self, new_facts):
        if new_facts:
            for fact in new_facts:
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
                self.facts[fact] = self.default_ttl

    def decay(self):
        expired = []
        for fact in self.facts:
            self.facts[fact] -= 1
            if self.facts[fact] <= 0:
                expired.append(fact)
        for fact in expired:
            del self.facts[fact]
