from system2_poc import System2Engine, Inventory, ComparisonEvent, Name
from phase5_agm import AGMModule

engine = System2Engine()
engine.reset()
agm = AGMModule()

print("--- Test 2: Mutually Exclusive Event Contradiction ---")
engine.declare(ComparisonEvent(arg0="e1", arg1="double", frozen=False))
engine.declare(ComparisonEvent(arg0="e1", arg1="half", frozen=False))
engine.run()

print("\n--- Halting System 2. Calling System 3 AGM ---")
agm.resolve_contradictions(engine)

print("\n--- Resuming System 2 ---")
engine.run()
