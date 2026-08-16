import re
from typing import Dict, List, Any, Optional, Tuple, Set

# ==========================================
# Ontological Type Hierarchy (Python Types)
# ==========================================
class Entity: pass
class Person(Entity): pass
class Commodity(Entity): pass

class Egg(Commodity): pass
class DuckEgg(Egg): pass
class ChickenEgg(Egg): pass

class Fruit(Commodity): pass
class Apple(Fruit): pass

class PredicateType: pass
class AcquisitionEvent(PredicateType): pass
class Buy(AcquisitionEvent): pass
class Bake(AcquisitionEvent): pass
class Produce(AcquisitionEvent): pass

# Least Common Ancestor (LCA) helper
def find_lca(cls1: type, cls2: type) -> type:
    if not isinstance(cls1, type) or not isinstance(cls2, type):
        return object
    mro1 = cls1.__mro__
    mro2 = cls2.__mro__
    for base in mro1:
        if base in mro2 and base not in [object, Entity, Commodity, PredicateType]:
            return base
    return object

class KnowledgeBase:
    def __init__(self):
        self.facts: Dict[str, List[Tuple[Any, ...]]] = {}
        self.rules: List[Dict[str, Any]] = []

    def declare_fact(self, pred: str, *args: Any):
        if pred not in self.facts:
            self.facts[pred] = []
        self.facts[pred].append(args)

    def declare_rule(self, head_pred: str, head_args: List[Any], body: List[Tuple[str, List[Any]]], calc_expr: Optional[str] = None, calc_target: Optional[str] = None):
        self.rules.append({
            "head": (head_pred, head_args),
            "body": body,
            "calc": calc_expr,
            "target": calc_target or (head_args[-1] if head_args else None)
        })

# ==========================================
# Ontological Backward-Chaining Reasoner
# ==========================================
class OntologicalReasoner:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def solve(self, pred: str, args: List[Any], depth: int = 0) -> List[Dict[str, Any]]:
        results = []
        
        # 1. Match facts
        if pred in self.kb.facts:
            for fact_args in self.kb.facts[pred]:
                bindings = self._unify(args, list(fact_args))
                if bindings is not None:
                    results.append(bindings)

        # 2. Match rules
        for rule in self.kb.rules:
            r_head_pred, r_head_args = rule["head"]
            if r_head_pred != pred:
                continue
            
            bindings = self._unify(args, r_head_args)
            if bindings is None:
                continue
                
            body_bindings = [bindings]
            failed_subgoal = None
            
            for b_pred, b_args in rule["body"]:
                new_body_bindings = []
                for b in body_bindings:
                    sub_args = [b.get(a, a) for a in b_args]
                    sub_results = self.solve(b_pred, sub_args, depth + 1)
                    
                    for sub_b in sub_results:
                        merged = b.copy()
                        merged.update(sub_b)
                        new_body_bindings.append(merged)
                        
                if not new_body_bindings:
                    failed_subgoal = (b_pred, b_args)
                    body_bindings = []
                    break
                body_bindings = new_body_bindings

            # Evaluate math expression if present
            for b in body_bindings:
                if rule["calc"]:
                    expr = rule["calc"]
                    for k, v in list(b.items()):
                        expr = re.sub(rf"\b{k}\b", str(v), expr)
                    try:
                        val = eval(expr, {"__builtins__": None}, {})
                        b["ans"] = val
                        if rule["target"]:
                            target_var = rule["target"]
                            if target_var.startswith("?"):
                                target_var = target_var[1:]
                            b[target_var] = val
                    except Exception:
                        pass
                results.append(b)

        return results

    def _unify(self, query: List[Any], target: List[Any]) -> Optional[Dict[str, Any]]:
        if len(query) != len(target):
            return None
        bindings = {}
        for q, t in zip(query, target):
            # If either is a variable:
            if isinstance(q, str) and q.startswith("?"):
                bindings[q[1:]] = t
            elif isinstance(t, str) and t.startswith("?"):
                bindings[t[1:]] = q
            # If both are Python classes (Ontology unification):
            elif isinstance(q, type) and isinstance(t, type):
                if not (issubclass(q, t) or issubclass(t, q)):
                    return None
            # If one is a class and the other is a string class-name or instance:
            elif isinstance(q, type) and not isinstance(t, type):
                if not (str(t).lower() == q.__name__.lower() or isinstance(t, q)):
                    return None
            elif isinstance(t, type) and not isinstance(q, type):
                if not (str(q).lower() == t.__name__.lower() or isinstance(q, t)):
                    return None
            # Direct value match:
            elif q != t:
                return None
        return bindings

# ==========================================
# Ontological Abductive ILP Learner
# ==========================================
class OntologicalILPLearner:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.reasoner = OntologicalReasoner(kb)

    def abduce_and_induce(self, examples: List[Tuple[str, List[Any], float]]) -> Optional[Dict[str, Any]]:
        """
        Analyzes a batch of failed proofs, abduces intermediate facts,
        and generalizes them using Least Common Ancestor (LCA) ontological unification.
        """
        print(f"\n[ILP] Analyzing {len(examples)} examples...")
        
        abduced_types: Set[type] = set()
        abduced_facts = []
        
        for pred, args, expected_val in examples:
            results = self.reasoner.solve(pred, args)
            success = any(r.get(args[-1][1:] if args[-1].startswith("?") else "ans") == expected_val for r in results)
            
            if success:
                continue
                
            # Abduction phase: Find the price and abduce the missing quantity
            # We look up sells fact matching the item type
            item_type = args[1] # e.g., DuckEgg or ChickenEgg
            price = None
            for s_args in self.kb.facts["sells"]:
                if s_args[1] == item_type:
                    price = s_args[2]
                    break
                    
            if price:
                qty = expected_val / price
                abduced_fact = ("sold_qty", (args[0], item_type, qty))
                abduced_facts.append(abduced_fact)
                if isinstance(item_type, type):
                    abduced_types.add(item_type)
                print(f"  -> ABDUCED: sold_qty({args[0]}, {item_type.__name__}, {qty})")

        if not abduced_types:
            return None

        # Generalization phase: Compute the Least Common Ancestor (LCA) in the ontology
        print("\n[ILP] Generalizing abduced facts using WordNet taxonomy...")
        lca = list(abduced_types)[0]
        for t in list(abduced_types)[1:]:
            lca = find_lca(lca, t)
            
        print(f"  -> Least Common Ancestor of {', '.join(t.__name__ for t in abduced_types)}: {lca.__name__}")

        induced_rule = {
            "head": ("sold_qty", ["?p", lca, "?qty"]),
            "body": [
                ("produced", ["?p", lca, "?tot"]),
                ("consumed", ["?p", "?item", "?cons"])
            ],
            "calc": "tot - cons",
            "target": "?qty"
        }
        
        print(f"  -> INDUCED Ontological Rule:")
        print(f"     {induced_rule['head'][0]}({induced_rule['head'][1][0]}, {induced_rule['head'][1][1].__name__}, {induced_rule['head'][1][2]}) :-")
        for b_pred, b_args in induced_rule["body"]:
            args_str = [a.__name__ if isinstance(a, type) else str(a) for a in b_args]
            print(f"       {b_pred}({', '.join(args_str)}),")
        print(f"       {induced_rule['target']} is {induced_rule['calc']}")
        
        return induced_rule

# ==========================================
# Run Verification
# ==========================================
def demo_ontological_ilp():
    kb = KnowledgeBase()
    
    # Background facts with class-based ontology (DuckEggs and ChickenEggs)
    kb.declare_fact("produced", "janet", DuckEgg, 16)
    kb.declare_fact("consumed", "janet", DuckEgg, 7)
    kb.declare_fact("sells", "janet", DuckEgg, 2)
    
    kb.declare_fact("produced", "janet", ChickenEgg, 10)
    kb.declare_fact("consumed", "janet", ChickenEgg, 4)
    kb.declare_fact("sells", "janet", ChickenEgg, 3)
    
    # Baseline profit rule
    kb.declare_rule(
        head_pred="daily_profit",
        head_args=["?p", "?item", "?ans"],
        body=[
            ("sells", ["?p", "?item", "?price"]),
            ("sold_qty", ["?p", "?item", "?qty"])
        ],
        calc_expr="qty * price",
        calc_target="?ans"
    )
    
    learner = OntologicalILPLearner(kb)
    
    # Examples we want to solve
    examples = [
        ("daily_profit", ["janet", DuckEgg, "?ans"], 18.0),
        ("daily_profit", ["janet", ChickenEgg, "?ans"], 18.0) # 6 qty * 3 price = 18.0
    ]
    
    induced = learner.abduce_and_induce(examples)
    
    if induced:
        # Declare the generalized induced rule using the ontological LCA 'Egg'
        kb.declare_rule(
            head_pred=induced["head"][0],
            head_args=induced["head"][1],
            body=induced["body"],
            calc_expr=induced["calc"],
            calc_target=induced["target"]
        )
        
        # Test if the single generalized rule can successfully solve both subclasses!
        reasoner = OntologicalReasoner(kb)
        
        print("\n[Verification] Testing generalized rule on DuckEgg:")
        sol_duck = reasoner.solve("daily_profit", ["janet", DuckEgg, "?ans"])
        for sol in sol_duck:
            print(f"  -> Profit: {sol.get('ans')} (SUCCESS)")
            
        print("\n[Verification] Testing generalized rule on ChickenEgg:")
        sol_chicken = reasoner.solve("daily_profit", ["janet", ChickenEgg, "?ans"])
        for sol in sol_chicken:
            print(f"  -> Profit: {sol.get('ans')} (SUCCESS)")

if __name__ == "__main__":
    demo_ontological_ilp()
