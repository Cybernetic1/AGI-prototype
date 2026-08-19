import collections
import collections.abc
collections.Mapping = collections.abc.Mapping
from experta import *

class ItemTaxSlab(Fact): pass
class Transaction(Fact): pass
class Invoice(Fact): pass

class GSTEngine(KnowledgeEngine):
    
    @Rule(
        AS.txn << Transaction(
            seller_state=MATCH.s_state, 
            buyer_state=MATCH.b_state, 
            item=MATCH.item_name, 
            amount=MATCH.amt,
            processed=False
        ),
        ItemTaxSlab(item=MATCH.item_name, rate=MATCH.rate),
        TEST(lambda s_state, b_state: s_state.lower() == b_state.lower())
    )
    def calc_intra_state(self, txn, s_state, item_name, amt, rate):
        """Intra-state (same state): Split tax into CGST and SGST"""
        print(f"\n[System 2: GST Engine] INTRA-STATE transaction detected in {s_state.title()}.")
        cgst = amt * (rate / 2)
        sgst = amt * (rate / 2)
        total = amt + cgst + sgst
        print(f"  -> Item: {item_name.title()} | Base Amount: Rs {amt} | GST Rate: {rate*100}%")
        print(f"  -> CGST (Central): Rs {cgst} | SGST (State): Rs {sgst} | IGST: Rs 0")
        print(f"  -> Total Invoice Value: Rs {total}\n")
        self.modify(txn, processed=True)
        self.declare(Invoice(item=item_name, base=amt, cgst=cgst, sgst=sgst, igst=0, total=total))


    @Rule(
        AS.txn << Transaction(
            seller_state=MATCH.s_state, 
            buyer_state=MATCH.b_state, 
            item=MATCH.item_name, 
            amount=MATCH.amt,
            processed=False
        ),
        ItemTaxSlab(item=MATCH.item_name, rate=MATCH.rate),
        TEST(lambda s_state, b_state: s_state.lower() != b_state.lower())
    )
    def calc_inter_state(self, txn, s_state, b_state, item_name, amt, rate):
        """Inter-state (cross-border): Full tax goes to IGST"""
        print(f"\n[System 2: GST Engine] INTER-STATE transaction detected: {s_state.title()} -> {b_state.title()}.")
        igst = amt * rate
        total = amt + igst
        print(f"  -> Item: {item_name.title()} | Base Amount: Rs {amt} | GST Rate: {rate*100}%")
        print(f"  -> CGST: Rs 0 | SGST: Rs 0 | IGST (Integrated): Rs {igst}")
        print(f"  -> Total Invoice Value: Rs {total}\n")
        self.modify(txn, processed=True)
        self.declare(Invoice(item=item_name, base=amt, cgst=0, sgst=0, igst=igst, total=total))

if __name__ == "__main__":
    engine = GSTEngine()
    engine.reset()
    
    # Preload the government tax slabs into System 2
    engine.declare(ItemTaxSlab(item="laptop", rate=0.18))
    engine.declare(ItemTaxSlab(item="rice", rate=0.05))
    engine.declare(ItemTaxSlab(item="software_service", rate=0.18))
    
    # Mocking System 1 outputs from Hinglish/Tanglish text:
    
    # "Bhai, 50000 ka laptop bhejna Delhi se Delhi"
    # System 1 parses to: Transaction(Delhi -> Delhi, laptop, 50000)
    engine.declare(Transaction(seller_state="Delhi", buyer_state="Delhi", item="laptop", amount=50000, processed=False))
    
    # "Customer Mumbai mein hai, hum Bangalore se 2000 ka software service de rahe hain"
    # System 1 parses to: Transaction(Karnataka -> Maharashtra, software_service, 2000)
    engine.declare(Transaction(seller_state="Karnataka", buyer_state="Maharashtra", item="software_service", amount=2000, processed=False))

    engine.run()
