from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .lorentz_math import project_to_hyperboloid, lorentz_distance
except ImportError:  # pragma: no cover - direct script execution fallback
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lorentz_math import project_to_hyperboloid, lorentz_distance

class LogicTransformer(nn.Module):
    """
    Logic Transformer (LT) / Differentiable Logic Network (DLN)

    Acts as System 1 in a Tri-System Neurosymbolic Architecture.
    This module applies fuzzy, differentiable logic rules in parallel over
    a continuous working memory using tensor operations (unification & cylindrification).
    """
    def __init__(self, prop_length, max_props, output_dim, num_rules=8, num_premises=2, var_slots=3):
        """
        Args:
            prop_length (L): Dimensionality of each semantic proposition.
            max_props (W): Maximum number of propositions in working memory.
            output_dim (D): Output dimension for the logical conclusions.
            num_rules (M): Number of parallel learned logic rules.
            num_premises (J): Number of premises per rule.
            var_slots (I): Number of logic variables available per rule.
        """
        super().__init__()
        self.M = num_rules
        self.J = num_premises
        self.I = var_slots
        self.L = prop_length
        self.W = max_props
        
        # Rule Constants: The latent fuzzy templates each premise matches against.
        self.premise_constants = nn.Parameter(torch.randn(num_rules, num_premises, prop_length) * 0.1)
        
        # Slot Selectors (Cylindrification): Attention logits to bind working memory propositions to variable slots.
        self.slot_selectors = nn.Parameter(torch.randn(num_rules, num_premises, var_slots, max_props) * 0.1)
        
        # Consequent Projections (Rule Heads): Maps the unified variables to the final conclusion vector.
        self.rule_weights = nn.Parameter(torch.randn(num_rules, num_premises * prop_length, output_dim) / (prop_length ** 0.5))
        self.rule_bias = nn.Parameter(torch.zeros(num_rules, output_dim))
        
    def forward(self, working_memory, temperature=1.0):
        """
        Args:
            working_memory: (Batch, W, L) tensor of semantic propositions.
            temperature: Controls the sharpness of fuzzy unification.
            
        Returns:
            outputs: (Batch, output_dim) tensor representing the summed logical conclusions.
        """
        B, W_in, L_in = working_memory.shape
        assert L_in == self.L, f"Expected proposition length {self.L}, got {L_in}"
        
        # Truncate or pad slot selectors if actual W differs from max_props
        W = min(W_in, self.W)
        wm_subset = working_memory[:, :W, :]
        
        # 1. Variable Binding / Cylindrification
        # Compute soft attention over the working memory propositions for each variable slot
        # slot_attention: (M, J, I, W)
        slot_attention = F.softmax(self.slot_selectors[..., :W] / temperature, dim=-1)
        
        # 2. Extract values for variables
        # Soft-select propositions from working memory: (B, M, J, I, L)
        selected_slots = torch.einsum('bwl,mjiw->bmjil', wm_subset, slot_attention)
        
        # Average over variable slots to form the unified premise representation
        # selected_avg: (B, M, J, L)
        selected_avg = selected_slots.mean(dim=3)
        
        # 3. Fuzzy Matching (Lorentz distance to rule constants)
        # diff: (B, M, J, L)
        try:
            from .lorentz_math import project_to_hyperboloid, lorentz_distance
        except ImportError:  # pragma: no cover - direct script execution fallback
            import sys
            from pathlib import Path

            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from lorentz_math import project_to_hyperboloid, lorentz_distance
        
        # Ensure both premise constants and selected embeddings are strictly on the hyperboloid
        hyper_constants = project_to_hyperboloid(self.premise_constants)
        hyper_selected = project_to_hyperboloid(selected_avg)
        
        # match_scores is the negative distance (higher is better match)
        # hyper_selected: (B, M, J, L)
        # hyper_constants: (1, M, J, L)
        distances = lorentz_distance(hyper_selected, hyper_constants.unsqueeze(0))
        match_scores = -distances
        
        # 4. Premise Attention
        # premise_attention: (B, M, J)
        premise_attention = F.softmax(match_scores / temperature, dim=-1)
        
        # 5. Modulate and project to conclusion
        # Weight the representations by how well they matched the rule conditions
        # weighted_selected: (B, M, J, L)
        weighted_selected = selected_avg * premise_attention.unsqueeze(-1)
        
        # Flatten premises to feed into the rule head: (B, M, J*L)
        rule_inputs = weighted_selected.view(B, self.M, self.J * self.L)
        
        # Rule outputs: (B, M, output_dim)
        rule_outputs = torch.einsum('bmk,mkd->bmd', rule_inputs, self.rule_weights) + self.rule_bias

        # 6. Aggregation
        # Sum rule outputs (OR-like aggregation) to form the final logical inference
        return rule_outputs.sum(dim=1)

    def _predicate_templates(self):
        """Canonical proposition templates used for the ratio-only LT probe."""
        return {
            "has": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            "ratio": torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            "query": torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        }

    def _map_premise_to_predicate(self, premise_vector, predicate_templates=None):
        if predicate_templates is None:
            predicate_templates = self._predicate_templates()
        premise_vector = premise_vector.detach().reshape(1, -1)
        best_name = None
        best_distance = float("inf")
        for name, template in predicate_templates.items():
            distance = torch.linalg.norm(premise_vector - template.reshape(1, -1)).item()
            if distance < best_distance:
                best_distance = distance
                best_name = name
        return best_name, best_distance

    def _slot_variables(self):
        return ["?A", "?B", "?C"]

    def _rule_to_symbolic_form(self, rule_idx, temp=1.0, tol=1e-3, predicate_templates=None):
        if predicate_templates is None:
            predicate_templates = self._predicate_templates()

        vars_ = self._slot_variables()
        premise_atoms = []
        for premise_idx in range(self.J):
            premise_vec = self.premise_constants[rule_idx, premise_idx]
            pred_name, _ = self._map_premise_to_predicate(premise_vec, predicate_templates)
            premise_atoms.append(f"{pred_name}({', '.join(vars_)})")

        conclusion = "Target(?, ?, ?item)"
        return f"({ ' ∧ '.join(premise_atoms) }) ⟹ {conclusion}"

    def explain_rules(self, temp=1.0, tol=1e-3, predicate_templates=None):
        """Render a compact symbolic summary of the learned rules in logic notation."""
        if predicate_templates is None:
            predicate_templates = self._predicate_templates()

        lines = []
        for rule_idx in range(self.M):
            rule_expr = self._rule_to_symbolic_form(rule_idx, temp=temp, tol=tol, predicate_templates=predicate_templates)
            lines.append(f"Rule {rule_idx}: {rule_expr}")
        return "\n".join(lines)

    def save_rules(self, path, temp=1.0, tol=1e-3, predicate_templates=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered = self.explain_rules(temp=temp, tol=tol, predicate_templates=predicate_templates)
        path.write_text(rendered + "\n", encoding="utf-8")
        return str(path)

    def print_rules(self, temp=1.0, tol=1e-3, predicate_templates=None):
        print(self.explain_rules(temp=temp, tol=tol, predicate_templates=predicate_templates))
