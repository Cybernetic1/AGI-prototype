import collections
import collections.abc
collections.Mapping = collections.abc.Mapping
from experta import *

from ontology import (
    ComparisonEvent,
    CreationEvent,
    LossEvent,
    Event as OntologyEvent,
    TransferEvent,
)

# =======================================================
# Neo-Davidsonian Semantic Classes
# =======================================================
Event = OntologyEvent
class Predicate(Fact): pass
class Agent(Fact): pass
class Recipient(Fact): pass
class Patient(Fact): pass
class Name(Fact): pass
class Quantity(Fact): pass
class Type(Fact): pass
class Tense(Fact): pass
class Entity(Fact): pass
class Location(Fact): pass
class Time(Fact): pass
class Modifier(Fact): pass

# =======================================================
# State and Belief Classes
# =======================================================
class Belief(Fact): pass
class Inventory(Fact): pass
class EventProcessed(Fact): pass
class Contradiction(Fact): pass
class Produced(Fact): pass
class Consumed(Fact): pass
class Sells(Fact): pass
class Rate(Fact): pass
class Has(Fact): pass
class Total(Fact): pass
class Remaining(Fact): pass
class ScaleFactor(Fact): pass
class Add(Fact): pass
class Subtract(Fact): pass
class Multiply(Fact): pass
class Divide(Fact): pass
class Percent(Fact): pass
class Ratio(Fact): pass
class UnitConvert(Fact): pass
class Query(Fact): pass

# =======================================================
# SYSTEM 2 ENGINE (Math & State Tracker)
# =======================================================
class System2Engine(KnowledgeEngine):
    
    # ----------------------------------------------------
    # PHASE 5: CONTRADICTION DETECTION & TMS
    # ----------------------------------------------------
    @Rule(
        AS.inv << Inventory(owner=MATCH.owner, item=MATCH.item, qty=MATCH.qty & P(lambda x: int(x) < 0), frozen=~MATCH.f)
    )
    def detect_negative_inventory(self, owner, item, qty, inv):
        """
        TMS Rule: Physical inventories cannot be negative.
        If a sequence of events results in a negative inventory, System 1 has hallucinated 
        an impossible transfer or loss, or the premises are contradictory.
        """
        print(f"\n[System 2 TMS] CONTRADICTION DETECTED: Negative Inventory")
        print(f"  -> {owner} has {qty} {item}(s), which is impossible.")
        print(f"  -> Escalating to System 3 for Belief Revision...\n")
        self.declare(Contradiction(
            type="negative_inventory", 
            owner=owner, 
            item=item, 
            qty=qty
        ))
        self.modify(inv, frozen=True)
        
    @Rule(
        AS.c1 << ComparisonEvent(arg0=MATCH.e, arg1=MATCH.verb1, frozen=False),
        AS.c2 << ComparisonEvent(arg0=MATCH.e, arg1=MATCH.verb2 & P(lambda x: x != MATCH.verb1), frozen=False),
        TEST(lambda verb1, verb2: verb1 < verb2) # Prevent permutation duplicates
    )
    def detect_mutually_exclusive_events(self, e, verb1, verb2, c1, c2):
        """
        TMS Rule: An event cannot simultaneously be two mutually exclusive relation types 
        (e.g., event e1 cannot be both 'double' and 'half').
        """
        print(f"\n[System 2 TMS] CONTRADICTION DETECTED: Mutually Exclusive Event Types")
        print(f"  -> Event {e} is classified as both '{verb1}' and '{verb2}'.")
        print(f"  -> Escalating to System 3 for Belief Revision...\n")
        self.declare(Contradiction(
            type="mutually_exclusive_event", 
            event=e, 
            conflict=(verb1, verb2)
        ))
        self.modify(c1, frozen=True)
        self.modify(c2, frozen=True)
    
    # ----------------------------------------------------
    # AXIOM 1: Transfer (Add/Subtract)
    # ----------------------------------------------------
    @Rule(
        TransferEvent(arg0=MATCH.e, arg1=MATCH.verb, frozen=~MATCH.f),
        NOT(EventProcessed(id=MATCH.e)),
        Agent(arg0=MATCH.e, arg1=MATCH.a_var),
        Name(arg0=MATCH.a_var, arg1=MATCH.giver),
        Recipient(arg0=MATCH.e, arg1=MATCH.r_var),
        Name(arg0=MATCH.r_var, arg1=MATCH.receiver),
        Patient(arg0=MATCH.e, arg1=MATCH.p_var),
        Name(arg0=MATCH.p_var, arg1=MATCH.item),
        Quantity(arg0=MATCH.p_var, arg1=MATCH.q_str),
        AS.giver_inv << Inventory(owner=MATCH.giver, item=MATCH.item, qty=MATCH.g_qty),
        AS.rec_inv << Inventory(owner=MATCH.receiver, item=MATCH.item, qty=MATCH.r_qty)
    )
    def execute_transfer(self, e, verb, giver, receiver, item, q_str, giver_inv, g_qty, rec_inv, r_qty):
        q = int(q_str)
        print(f"\n[System 2] AXIOM 1 TRIGGERED: Transfer")
        print(f"  -> Event {e}: {giver} {verb}s {q} {item}(s) to {receiver}.")
        print(f"  -> {giver}'s inventory: {g_qty} - {q} = {g_qty - q}")
        print(f"  -> {receiver}'s inventory: {r_qty} + {q} = {r_qty + q}\n")
        
        self.modify(giver_inv, qty=g_qty - q)
        self.modify(rec_inv, qty=r_qty + q)
        self.declare(EventProcessed(id=e))

    # ----------------------------------------------------
    # AXIOM 2: Generation/Creation (Add)
    # ----------------------------------------------------
    @Rule(
        CreationEvent(arg0=MATCH.e, arg1=MATCH.verb, frozen=~MATCH.f),
        NOT(EventProcessed(id=MATCH.e)),
        Agent(arg0=MATCH.e, arg1=MATCH.a_var),
        Name(arg0=MATCH.a_var, arg1=MATCH.agent),
        Patient(arg0=MATCH.e, arg1=MATCH.p_var),
        Name(arg0=MATCH.p_var, arg1=MATCH.item),
        Quantity(arg0=MATCH.p_var, arg1=MATCH.q_str),
        AS.agent_inv << Inventory(owner=MATCH.agent, item=MATCH.item, qty=MATCH.a_qty)
    )
    def execute_generation(self, e, verb, agent, item, q_str, agent_inv, a_qty):
        q = int(q_str)
        print(f"\n[System 2] AXIOM 2 TRIGGERED: Generation")
        print(f"  -> Event {e}: {agent} {verb}s {q} {item}(s).")
        print(f"  -> {agent}'s inventory: {a_qty} + {q} = {a_qty + q}\n")
        
        self.modify(agent_inv, qty=a_qty + q)
        self.declare(EventProcessed(id=e))

    # ----------------------------------------------------
    # AXIOM 3: Destruction/Loss (Subtract)
    # ----------------------------------------------------
    @Rule(
        LossEvent(arg0=MATCH.e, arg1=MATCH.verb, frozen=~MATCH.f),
        NOT(EventProcessed(id=MATCH.e)),
        Agent(arg0=MATCH.e, arg1=MATCH.a_var),
        Name(arg0=MATCH.a_var, arg1=MATCH.agent),
        Patient(arg0=MATCH.e, arg1=MATCH.p_var),
        Name(arg0=MATCH.p_var, arg1=MATCH.item),
        Quantity(arg0=MATCH.p_var, arg1=MATCH.q_str),
        AS.agent_inv << Inventory(owner=MATCH.agent, item=MATCH.item, qty=MATCH.a_qty)
    )
    def execute_destruction(self, e, verb, agent, item, q_str, agent_inv, a_qty):
        q = int(q_str)
        print(f"\n[System 2] AXIOM 3 TRIGGERED: Destruction")
        print(f"  -> Event {e}: {agent} {verb}s {q} {item}(s).")
        print(f"  -> {agent}'s inventory: {a_qty} - {q} = {a_qty - q}\n")
        
        self.modify(agent_inv, qty=a_qty - q)
        self.declare(EventProcessed(id=e))

    # ----------------------------------------------------
    # AXIOM 4: Comparison/Relational (Multiply/Divide)
    # ----------------------------------------------------
    @Rule(
        # Matches logic like "Mary has twice as many apples as John"
        ComparisonEvent(arg0=MATCH.e, arg1=MATCH.verb, frozen=~MATCH.f),
        NOT(EventProcessed(id=MATCH.e)),
        Agent(arg0=MATCH.e, arg1=MATCH.a_var),
        Name(arg0=MATCH.a_var, arg1=MATCH.agent),
        Patient(arg0=MATCH.e, arg1=MATCH.p_var),
        Name(arg0=MATCH.p_var, arg1=MATCH.item),
        Modifier(arg0=MATCH.e, arg1=MATCH.multiplier & P(lambda x: x in ["twice", "double", "triple", "half"])),
        AS.agent_inv << Inventory(owner=MATCH.agent, item=MATCH.item, qty=MATCH.a_qty)
    )
    def execute_multiplier(self, e, agent, item, multiplier, agent_inv, a_qty):
        # NOTE: Simplified for now since spaCy's default logic doesn't cleanly separate "reference"
        # without custom LT training. We will rely on LT to emit a Multiplier fact.
        print(f"\n[System 2] AXIOM 4 TRIGGERED: Multiplier")
        
        mult_val = 2 if multiplier in ["twice", "double"] else 3 if multiplier == "triple" else 0.5
        # q = int(ref_qty * mult_val)
        
        # print(f"  -> Event {e}: {agent} has {multiplier} as many {item}(s) as {reference}.")
        # print(f"  -> {agent}'s inventory: {a_qty} -> {q}\n")
        
        # self.modify(agent_inv, qty=q)
        self.declare(EventProcessed(id=e))

    @Rule(
        AS.p << Produced(owner=MATCH.owner, item=MATCH.item, qty=MATCH.produced),
        AS.c << Consumed(owner=MATCH.owner, item=MATCH.item, qty=MATCH.consumed),
        AS.s << Sells(owner=MATCH.owner, item=MATCH.item, price=MATCH.price),
        TEST(lambda produced, consumed, price: float(produced) >= float(consumed))
    )
    def compute_profit_from_ledger(self, owner, item, produced, consumed, price, p, c, s):
        sold = float(produced) - float(consumed)
        value = sold * float(price)
        print(f"\n[System 2] Ledger Profit Rule: {owner} sold {sold} {item}(s) at ${price} each => ${value}")
        self.declare(Belief(label="daily_profit", value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.p << Produced(owner=MATCH.owner, item=MATCH.item, qty=MATCH.produced),
        AS.c << Consumed(owner=MATCH.owner, item=MATCH.item, qty=MATCH.consumed),
        TEST(lambda goal: goal in {"total_quantity", "remaining_quantity", "sold_quantity"})
    )
    def compute_quantity_from_ledger(self, goal, owner, item, produced, consumed, q, p, c):
        if goal == "total_quantity":
            value = float(produced) + float(consumed)
            label = "total_quantity"
        elif goal == "remaining_quantity":
            value = float(produced) - float(consumed)
            label = "remaining_quantity"
        else:
            value = float(produced) - float(consumed)
            label = "sold_quantity"
        print(f"\n[System 2] Quantity Rule: {label} for {owner} = {value}")
        self.declare(Belief(label=label, value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.scale << ScaleFactor(owner=MATCH.owner, item=MATCH.item, factor=MATCH.factor),
        AS.base << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.qty),
        TEST(lambda goal: goal in {"scaled_quantity", "total_quantity"})
    )
    def compute_scaled_quantity(self, goal, owner, item, factor, qty, q, scale, base):
        value = float(qty) * float(factor)
        label = "scaled_quantity" if goal == "scaled_quantity" else "total_quantity"
        print(f"\n[System 2] Scale Rule: {label} for {owner}/{item} = {value}")
        self.declare(Belief(label=label, value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.h << Has(owner=MATCH.owner, item=MATCH.item, qty=MATCH.qty),
        TEST(lambda goal: goal == "total_quantity")
    )
    def compute_total_from_has(self, goal, owner, item, qty, q, h):
        value = float(qty)
        print(f"\n[System 2] Has Rule: total_quantity for {owner}/{item} = {value}")
        self.declare(Belief(label="total_quantity", value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.r << Rate(item=MATCH.item, price_per_unit=MATCH.price),
        AS.s << Produced(owner=MATCH.owner, item=MATCH.item, qty=MATCH.qty),
        TEST(lambda goal: goal == "value_of_produced")
    )
    def compute_value_of_produced(self, goal, owner, item, price, qty, q, r, s):
        value = float(price) * float(qty)
        print(f"\n[System 2] Rate Rule: value of {qty} {item} for {owner} = {value}")
        self.declare(Belief(label="value_of_produced", value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.a << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.a_qty),
        AS.b << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.b_qty),
        TEST(lambda goal: goal == "sum_quantities")
    )
    def sum_quantities(self, goal, owner, item, a_qty, b_qty, q, a, b):
        value = float(a_qty) + float(b_qty)
        print(f"\n[System 2] Sum Rule: {owner}/{item} total = {value}")
        self.declare(Belief(label="sum_quantities", value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.a << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.a_qty),
        AS.b << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.b_qty),
        TEST(lambda goal: goal == "difference_quantities")
    )
    def difference_quantities(self, goal, owner, item, a_qty, b_qty, q, a, b):
        value = float(a_qty) - float(b_qty)
        print(f"\n[System 2] Difference Rule: {owner}/{item} delta = {value}")
        self.declare(Belief(label="difference_quantities", value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.r << Rate(item=MATCH.item, price_per_unit=MATCH.price),
        AS.n << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.n_qty),
        TEST(lambda goal: goal == "value_per_unit")
    )
    def compute_value_per_unit(self, goal, owner, item, price, n_qty, q, r, n):
        value = float(price) * float(n_qty)
        print(f"\n[System 2] Unit Value Rule: {owner} has {n_qty} {item} at ${price} each => ${value}")
        self.declare(Belief(label="value_per_unit", value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.p << Produced(owner=MATCH.owner, item=MATCH.item, qty=MATCH.produced),
        AS.c << Consumed(owner=MATCH.owner, item=MATCH.item, qty=MATCH.consumed),
        TEST(lambda goal: goal == "remaining_quantity")
    )
    def compute_remaining(self, goal, owner, item, produced, consumed, q, p, c):
        value = float(produced) - float(consumed)
        print(f"\n[System 2] Remaining Rule: {owner}/{item} left = {value}")
        self.declare(Belief(label="remaining_quantity", value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.total << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.total_qty),
        AS.used << Consumed(owner=MATCH.owner, item=MATCH.item, qty=MATCH.used_qty),
        TEST(lambda goal: goal == "remaining_quantity")
    )
    def compute_remaining_from_quantity(self, goal, owner, item, total_qty, used_qty, q, total, used):
        value = float(total_qty) - float(used_qty)
        print(f"\n[System 2] Remaining-From-Quantity Rule: {owner}/{item} left = {value}")
        self.declare(Belief(label="remaining_quantity", value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.base << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.base_qty),
        AS.p << Percent(owner=MATCH.owner, item=MATCH.item, pct=MATCH.pct),
        TEST(lambda goal: goal in {"percent_increase", "discount_amount", "final_price"})
    )
    def percent_increase(self, goal, owner, item, base_qty, pct, q, base, p):
        if goal == "final_price":
            value = float(base_qty) * (1.0 - float(pct))
            label = "final_price"
        elif goal == "discount_amount":
            value = float(base_qty) * float(pct)
            label = "discount_amount"
        else:
            value = float(base_qty) * float(pct)
            label = "percent_increase"
        print(f"\n[System 2] Percent Rule: {owner}/{item} goal={goal} => {value}")
        self.declare(Belief(label=label, value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.r << Ratio(owner=MATCH.owner, item=MATCH.item, factor=MATCH.factor),
        AS.base << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.base_qty),
        TEST(lambda goal: goal in {"ratio_compose", "final_quantity"})
    )
    def ratio_compose(self, goal, owner, item, factor, base_qty, q, r, base):
        value = float(base_qty) * float(factor)
        print(f"\n[System 2] Ratio Rule: composed ratio result = {value}")
        self.declare(Belief(label=goal, value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.r << Ratio(owner=MATCH.owner, item=MATCH.item, factor=MATCH.factor, base_owner=MATCH.base_owner),
        AS.base << Quantity(owner=MATCH.base_owner, item=MATCH.item, qty=MATCH.base_qty),
        TEST(lambda goal: goal in {"ratio_compose", "final_quantity"})
    )
    def ratio_compose_reference(self, goal, owner, item, factor, base_owner, base_qty, q, r, base):
        value = float(base_qty) * float(factor)
        print(f"\n[System 2] Ratio Rule (ref-owner): {owner} = {factor} × {base_owner}/{item} = {value}")
        self.declare(Belief(label=goal, value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.total << Total(owner=MATCH.owner, item=MATCH.item, qty=MATCH.total_qty),
        AS.used << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.used_qty),
        TEST(lambda goal: goal == "remaining_quantity")
    )
    def remaining_quantity_from_total(self, goal, owner, item, total_qty, used_qty, q, total, used):
        value = float(total_qty) - float(used_qty)
        print(f"\n[System 2] Remaining Rule: {owner}/{item} left = {value}")
        self.declare(Belief(label="remaining_quantity", value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.rate << Rate(item=MATCH.item, price_per_unit=MATCH.price_per_unit),
        AS.hours << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.hours_qty),
        TEST(lambda goal: goal in {"rate_times_duration", "final_speed"})
    )
    def rate_times_duration_generic(self, goal, owner, item, price_per_unit, hours_qty, q, rate, hours):
        value = float(price_per_unit) * float(hours_qty)
        print(f"\n[System 2] Generic Rate-Time Rule: {value}")
        self.declare(Belief(label=goal, value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.rate << Rate(item=MATCH.item, price_per_unit=MATCH.price),
        AS.pct << Percent(owner=MATCH.owner, item=MATCH.item, pct=MATCH.pct),
        AS.qty << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.qty),
        TEST(lambda goal: goal == "alternate_price_total")
    )
    def alternate_price_total(self, goal, owner, item, price, pct, qty, q, rate, p, qty_fact):
        value = float(qty) * float(price) * (1.0 + float(pct)) / 2.0
        print(f"\n[System 2] Alternate Price Rule: {owner}/{item} total = {value}")
        self.declare(Belief(label="alternate_price_total", value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.rate << Rate(item=MATCH.item, price_per_unit=MATCH.price),
        AS.duration << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.duration_qty),
        TEST(lambda goal: goal == "rate_times_duration")
    )
    def rate_times_duration(self, goal, owner, item, price, duration_qty, q, rate, duration):
        value = float(price) * float(duration_qty)
        print(f"\n[System 2] Rate-Time Rule: {owner}/{item} over time = {value}")
        self.declare(Belief(label="rate_times_duration", value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.add << Add(lhs=MATCH.lhs, rhs=MATCH.rhs, result=MATCH.result),
        TEST(lambda goal: goal == "add_equation")
    )
    def add_equation(self, goal, lhs, rhs, result, q, add):
        value = float(lhs) + float(rhs)
        print(f"\n[System 2] Add Equation: {lhs} + {rhs} = {value}")
        self.declare(Belief(label="add_equation", value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.u << UnitConvert(value=MATCH.value, from_unit=MATCH.from_unit, to_unit=MATCH.to_unit, result=MATCH.result),
        TEST(lambda goal: goal == "unit_convert")
    )
    def unit_convert(self, goal, value, from_unit, to_unit, result, q, u):
        converted = float(value)
        print(f"\n[System 2] Unit Convert Rule: {value} {from_unit} -> {converted} {to_unit}")
        self.declare(Belief(label="unit_convert", value=converted, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.a << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.a_qty),
        AS.b << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.b_qty),
        TEST(lambda goal: goal == "add")
    )
    def add_explicit(self, goal, owner, item, a_qty, b_qty, q, a, b):
        value = float(a_qty) + float(b_qty)
        print(f"\n[System 2] Explicit Add Rule: {owner}/{item} = {value}")
        self.declare(Belief(label="add", value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.a << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.a_qty),
        AS.b << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.b_qty),
        TEST(lambda goal: goal == "subtract")
    )
    def subtract_explicit(self, goal, owner, item, a_qty, b_qty, q, a, b):
        value = float(a_qty) - float(b_qty)
        print(f"\n[System 2] Explicit Subtract Rule: {owner}/{item} = {value}")
        self.declare(Belief(label="subtract", value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.a << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.a_qty),
        AS.b << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.b_qty),
        TEST(lambda goal: goal == "multiply")
    )
    def multiply_explicit(self, goal, owner, item, a_qty, b_qty, q, a, b):
        value = float(a_qty) * float(b_qty)
        print(f"\n[System 2] Explicit Multiply Rule: {owner}/{item} = {value}")
        self.declare(Belief(label="multiply", value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.a << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.a_qty),
        AS.b << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.b_qty),
        TEST(lambda goal: goal == "divide")
    )
    def divide_explicit(self, goal, owner, item, a_qty, b_qty, q, a, b):
        value = float(a_qty) / float(b_qty)
        print(f"\n[System 2] Explicit Divide Rule: {owner}/{item} = {value}")
        self.declare(Belief(label="divide", value=value, activation=1.0))

    # ------------------------------------------------------------------
    # COMPOSITION RULES: local symbolic algebra for semantic micro-steps.
    # These are intentionally small and compositional: a base quantity with a
    # modifier (scale, rate, ratio, percent, unit conversion) produces an
    # intermediate `Multiply`/`Divide`/`Add`/`Subtract` fact, then a final
    # Belief. This keeps the logic general enough to support several GSM8K
    # patterns while staying short and easy to debug.
    # ------------------------------------------------------------------
    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.scale << ScaleFactor(owner=MATCH.owner, item=MATCH.item, factor=MATCH.factor),
        AS.base << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.base_qty),
        TEST(lambda goal: goal in {"compose_scale", "scaled_quantity"})
    )
    def compose_scale(self, goal, owner, item, factor, base_qty, q, scale, base):
        value = float(base_qty) * float(factor)
        print(f"\n[System 2] Composition Rule: scale {factor} * {base_qty} = {value}")
        self.declare(Multiply(lhs=base_qty, rhs=factor, result=value))
        self.declare(Belief(label=goal, value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.pct << Percent(owner=MATCH.owner, item=MATCH.item, pct=MATCH.pct),
        AS.base << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.base_qty),
        TEST(lambda goal: goal in {"compose_percent", "percent_increase"})
    )
    def compose_percent(self, goal, owner, item, pct, base_qty, q, pct_fact, base):
        value = float(base_qty) * float(pct)
        print(f"\n[System 2] Composition Rule: percent {pct} * {base_qty} = {value}")
        self.declare(Multiply(lhs=base_qty, rhs=pct, result=value))
        self.declare(Belief(label=goal, value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.ratio << Ratio(owner=MATCH.owner, item=MATCH.item, factor=MATCH.factor),
        AS.base << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.base_qty),
        TEST(lambda goal: goal in {"compose_ratio", "ratio_compose"})
    )
    def compose_ratio(self, goal, owner, item, factor, base_qty, q, ratio, base):
        value = float(base_qty) * float(factor)
        print(f"\n[System 2] Composition Rule: ratio {factor} * {base_qty} = {value}")
        self.declare(Multiply(lhs=base_qty, rhs=factor, result=value))
        self.declare(Belief(label=goal, value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.rate << Rate(item=MATCH.item, price_per_unit=MATCH.price_per_unit),
        AS.qty << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.qty),
        TEST(lambda goal: goal in {"compose_rate", "value_per_unit", "rate_times_duration"})
    )
    def compose_rate(self, goal, owner, item, price_per_unit, qty, q, rate, qty_fact):
        value = float(price_per_unit) * float(qty)
        print(f"\n[System 2] Composition Rule: rate {price_per_unit} * {qty} = {value}")
        self.declare(Multiply(lhs=price_per_unit, rhs=qty, result=value))
        self.declare(Belief(label=goal, value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.conv << UnitConvert(value=MATCH.value, from_unit=MATCH.from_unit, to_unit=MATCH.to_unit, result=MATCH.result),
        TEST(lambda goal: goal in {"compose_unit_convert", "unit_convert"})
    )
    def compose_unit_convert(self, goal, value, from_unit, to_unit, result, q, conv):
        print(f"\n[System 2] Composition Rule: unit convert {value} {from_unit} -> {result} {to_unit}")
        self.declare(Belief(label=goal, value=result, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.a << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.a_qty),
        AS.b << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.b_qty),
        TEST(lambda goal: goal in {"compose_add", "add", "sum_quantities"})
    )
    def compose_add(self, goal, owner, item, a_qty, b_qty, q, a, b):
        value = float(a_qty) + float(b_qty)
        print(f"\n[System 2] Composition Rule: add {a_qty} + {b_qty} = {value}")
        self.declare(Add(lhs=a_qty, rhs=b_qty, result=value))
        self.declare(Belief(label=goal, value=value, activation=1.0))

    @Rule(
        AS.q << Query(goal=MATCH.goal),
        AS.a << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.a_qty),
        AS.b << Quantity(owner=MATCH.owner, item=MATCH.item, qty=MATCH.b_qty),
        TEST(lambda goal: goal in {"compose_subtract", "subtract", "difference_quantities"})
    )
    def compose_subtract(self, goal, owner, item, a_qty, b_qty, q, a, b):
        value = float(a_qty) - float(b_qty)
        print(f"\n[System 2] Composition Rule: subtract {a_qty} - {b_qty} = {value}")
        self.declare(Subtract(lhs=a_qty, rhs=b_qty, result=value))
        self.declare(Belief(label=goal, value=value, activation=1.0))

if __name__ == "__main__":
    print("System 2 Axioms Ready.")
