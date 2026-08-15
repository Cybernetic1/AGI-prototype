import collections
import collections.abc
collections.Mapping = collections.abc.Mapping
from experta import *

# =======================================================
# ONTOLOGY DEFINITION (Hierarchical via Class Inheritance)
# =======================================================

class Entity(Fact):
    """Base class for all physical things"""
    pass

class Animal(Entity):
    """Any animal"""
    pass

class Dog(Animal):
    """A specific type of animal"""
    pass

class Person(Entity):
    """A human entity"""
    pass

class Perception(Fact):
    """Sensory input received from System 1 (Neural/Fast)"""
    pass

class Belief(Fact):
    """A deduced System 2 fact with a probability/certainty factor"""
    pass

# =======================================================
# SYSTEM 2 ENGINE
# =======================================================

class System2Engine(KnowledgeEngine):
    # LHS: Ontology Match
    # This rule looks for ANY Animal. Because Dog inherits from Animal, 
    # Experta will automatically match the Dog fact!
    @Rule(
        Animal(state="barking", prob=MATCH.p & P(lambda x: x > 0.8))
    )
    def react_to_loud_animal(self, p):
        print(f"[System 2] FAST REACTION: A loud animal is nearby! (Confidence: {p:.2f})")
        self.declare(Belief(label="distraction_present", prob=p))

    # LHS: Match only if probability thresholds are met using Python lambdas
    @Rule(
        Person(state="walking", prob=MATCH.p1 & P(lambda x: x > 0.7)),
        Perception(label="motion_detected", prob=MATCH.p2 & P(lambda x: x > 0.5))
    )
    def deduce_human_activity(self, p1, p2):
        # RHS: Execute Python code to calculate rudimentary joint probability
        confidence = p1 * p2
        print(f"[System 2] DEDUCED: Human activity confirmed via motion. (Confidence: {confidence:.2f})")
        self.declare(Belief(label="human_activity", prob=confidence))

    # LHS: React to the newly created Belief
    @Rule(
        Belief(label="human_activity", prob=MATCH.p & P(lambda x: x > 0.5)),
        Belief(label="distraction_present", prob=MATCH.p2)
    )
    def escalate_to_system3(self, p, p2):
        print(f"[System 2] -> ESCALATING to System 3: Complex social scene detected (Human + Distraction). Requires deliberative analysis.")
        # Here you would push this data to a queue for ProbLog / System 3

if __name__ == "__main__":
    print("--- Starting System 2 Cognitive Engine ---")
    engine = System2Engine()
    engine.reset()  # Prepares the engine
    
    # Simulated input coming from System 1
    # Notice we are asserting a 'Dog', not an 'Animal'
    engine.declare(Dog(name="Fido", state="barking", prob=0.90))
    engine.declare(Person(name="Alice", state="walking", prob=0.99))
    engine.declare(Perception(label="motion_detected", prob=0.60))
    
    engine.run()    # Fires the rules
    print("--- Execution Complete ---")
