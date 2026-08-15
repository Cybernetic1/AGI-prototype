import sys
import collections
import collections.abc
collections.Mapping = collections.abc.Mapping
from experta import *
import json

class ItemTaxSlab(Fact): pass
class Transaction(Fact): pass
class Invoice(Fact): pass
class VocabMapping(Fact): pass

class GSTInteractiveEngine(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self.invoice_generated = False
        
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
        print(f"\n✅ INTRA-STATE transaction detected in {s_state.title()}.")
        cgst = amt * (rate / 2)
        sgst = amt * (rate / 2)
        total = amt + cgst + sgst
        print(f"  -> CGST: Rs {cgst} | SGST: Rs {sgst} | IGST: Rs 0")
        print(f"  -> Total Invoice Value: Rs {total}\n")
        self.modify(txn, processed=True)
        self.invoice_generated = True

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
        print(f"\n✅ INTER-STATE transaction detected: {s_state.title()} -> {b_state.title()}.")
        igst = amt * rate
        total = amt + igst
        print(f"  -> CGST: Rs 0 | SGST: Rs 0 | IGST (Integrated): Rs {igst}")
        print(f"  -> Total Invoice Value: Rs {total}\n")
        self.modify(txn, processed=True)
        self.invoice_generated = True


class InteractiveChat:
    def __init__(self):
        self.engine = GSTInteractiveEngine()
        self.engine.reset()
        
        # Load initial knowledge
        self.engine.declare(ItemTaxSlab(item="laptop", rate=0.18))
        self.engine.declare(ItemTaxSlab(item="rice", rate=0.05))
        
        # Dictionary acting as our "collective knowledgebase" mapping dialects to canonical logic
        # e.g., "bhejna": "transfer", "pachas hazar": "50000"
        self.dictionary = {
            "bhejna": "transfer",
            "50k": "50000",
            "chawal": "rice"
        }
        self.business_state = "Delhi"

    def naive_parse(self, text):
        """Extremely naive parser just to simulate System 1 applying the dictionary"""
        words = text.lower().replace(",", "").replace(".", "").split()
        
        # Translate dialect
        canon_words = [self.dictionary.get(w, w) for w in words]
        
        # Extract features (hardcoded brute-force for POC)
        item = None
        amt = None
        buyer_state = None
        
        for w in canon_words:
            if w in ["laptop", "rice"]:
                item = w
            if w.isdigit():
                amt = float(w)
            if w in ["delhi", "mumbai", "maharashtra", "karnataka", "bangalore"]:
                buyer_state = "maharashtra" if w in ["mumbai", "maharashtra"] else "delhi"
                
        return item, amt, buyer_state

    def chat_loop(self):
        print("==================================================")
        print("🤖 AI GST Assistant (Hinglish/Tanglish Beta)")
        print("Type your transaction. Type 'quit' to exit.")
        print("==================================================")
        
        while True:
            text = input("\nUser > ")
            if text.lower() == 'quit':
                break
                
            item, amt, buyer_state = self.naive_parse(text)
            
            if item and amt and buyer_state:
                print(f"AI > Got it. Selling {item} to {buyer_state.title()} for Rs {amt}.")
                self.engine.invoice_generated = False
                self.engine.declare(Transaction(
                    seller_state=self.business_state,
                    buyer_state=buyer_state,
                    item=item,
                    amount=amt,
                    processed=False
                ))
                self.engine.run()
                
                if not self.engine.invoice_generated:
                    print(f"AI > I don't know the tax slab for '{item}'. Can you tell me the rate?")
            else:
                print("AI > I couldn't understand all the details. Let's learn!")
                if not amt:
                    ans = input("AI > What was the amount in that sentence? (e.g. '50000')\nUser > ")
                    print(f"AI > Learning mapping... How did you say {ans} in the sentence?")
                    word = input("User > ")
                    self.dictionary[word.lower()] = ans
                    print(f"AI > Learned! '{word}' -> '{ans}'. Try your sentence again.")
                elif not item:
                    ans = input("AI > What was the item you are selling?\nUser > ")
                    print(f"AI > Learning mapping... How did you say {ans} in the sentence?")
                    word = input("User > ")
                    self.dictionary[word.lower()] = ans
                    print(f"AI > Learned! '{word}' -> '{ans}'. Try your sentence again.")

if __name__ == "__main__":
    app = InteractiveChat()
    app.chat_loop()
