from system2_poc import System2Engine, Contradiction, Inventory, ComparisonEvent

class AGMModule:
    """
    Placeholder System 3 AGM module. 
    Observes contradictions emitted by System 2.
    """
    def resolve_contradictions(self, engine: System2Engine):
        """
        Reads all Contradiction facts from the engine and resolves them via 
        simulated epistemic entrenchment (probability/confidence).
        """
        contradictions = [f for f in engine.facts.values() if type(f).__name__ == "Contradiction"]
        
        for c in contradictions:
            c_type = c.get("type")
            print(f"\n[System 3 AGM] Analyzing {c_type}...")
            
            if c_type == "mutually_exclusive_event":
                # Find the frozen events
                event_id = c.get("event")
                conflict = c.get("conflict")
                frozen_facts = [f for f in engine.facts.values() 
                               if type(f).__name__ == "ComparisonEvent" 
                               and f.get("arg0") == event_id 
                               and f.get("frozen") == True]
                
                # Simulate Epistemic Entrenchment: pick one to keep, retract the other
                print(f"  -> Conflict: {conflict}. Assuming '{conflict[0]}' has higher prior probability.")
                print(f"  -> Action: Contraction of belief '{conflict[1]}'")
                
                for f in frozen_facts:
                    if f.get("arg1") == conflict[1]:
                        print(f"  -> Retracting Fact: {f}")
                        engine.retract(f)
                    else:
                        print(f"  -> Unfreezing Fact: {f}")
                        engine.modify(f, frozen=False)
                        
            elif c_type == "negative_inventory":
                owner = c.get("owner")
                print(f"  -> State impossible for {owner}. Assuming recent transaction was hallucinated.")
                print(f"  -> Action: Contraction. (Rollback state not fully implemented in this script)")
                
                # In a real TMS, we would track the justification graph (JTMS) 
                # and retract the specific Action/Event that caused this state change.
                # For now, we unfreeze the inventory and reset it to 0.
                inv_facts = [f for f in engine.facts.values() 
                             if type(f).__name__ == "Inventory" 
                             and f.get("owner") == owner 
                             and f.get("frozen") == True]
                
                for f in inv_facts:
                    print(f"  -> Resetting inventory to 0 and unfreezing.")
                    engine.modify(f, qty=0, frozen=False)

        # Retract the contradiction alerts so they don't fire again
        for c in contradictions:
            engine.retract(c)
