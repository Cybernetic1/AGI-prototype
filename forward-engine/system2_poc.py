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

# =======================================================
# SYSTEM 2 ENGINE (Math & State Tracker)
# =======================================================
class System2Engine(KnowledgeEngine):
    
    # ----------------------------------------------------
    # PHASE 5: CONTRADICTION DETECTION & TMS
    # ----------------------------------------------------
    @Rule(
        AS.inv << Inventory(owner=MATCH.owner, item=MATCH.item, qty=MATCH.qty & P(lambda x: int(x) < 0))
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
        
    @Rule(
        AS.c1 << ComparisonEvent(arg0=MATCH.e, arg1=MATCH.verb1),
        AS.c2 << ComparisonEvent(arg0=MATCH.e, arg1=MATCH.verb2 & P(lambda x: x != MATCH.verb1)),
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
    
    # ----------------------------------------------------
    # AXIOM 1: Transfer (Add/Subtract)
    # ----------------------------------------------------
    @Rule(
        TransferEvent(arg0=MATCH.e, arg1=MATCH.verb),
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
        CreationEvent(arg0=MATCH.e, arg1=MATCH.verb),
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
        LossEvent(arg0=MATCH.e, arg1=MATCH.verb),
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
        ComparisonEvent(arg0=MATCH.e, arg1=MATCH.verb),
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

if __name__ == "__main__":
    print("System 2 Axioms Ready.")
