from system2_poc import System2Engine, Inventory, ComparisonEvent, Name

engine = System2Engine()
engine.reset()

print("--- Test 1: Negative Inventory Contradiction ---")
engine.declare(Inventory(owner="John", item="apple", qty="-2"))
engine.run()

print("--- Test 2: Mutually Exclusive Event Contradiction ---")
engine.declare(ComparisonEvent(arg0="e1", arg1="double"))
engine.declare(ComparisonEvent(arg0="e1", arg1="half"))
engine.run()
