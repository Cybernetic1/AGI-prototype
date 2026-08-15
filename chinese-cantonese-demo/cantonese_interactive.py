import sys
import collections
import collections.abc
collections.Mapping = collections.abc.Mapping
from experta import *
import json

class TranslationRequest(Fact): pass
class ValidTranslation(Fact): pass
class MissingMapping(Fact): pass

class TranslationEngine(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self.translation_done = False
        
    @Rule(
        AS.req << TranslationRequest(
            mandarin=MATCH.mandarin,
            cantonese=MATCH.cantonese,
            processed=False
        ),
        TEST(lambda cantonese: cantonese is not None)
    )
    def output_translation(self, req, mandarin, cantonese):
        print(f"\n✅ SYSTEM 2 (Translation Engine): Valid mapping found.")
        print(f"  -> Mandarin Input: {mandarin}")
        print(f"  -> Cantonese Output: {cantonese}\n")
        self.modify(req, processed=True)
        self.declare(ValidTranslation(mandarin=mandarin, cantonese=cantonese))
        self.translation_done = True

    @Rule(
        AS.req << TranslationRequest(
            mandarin=MATCH.mandarin,
            cantonese=MATCH.cantonese,
            processed=False
        ),
        TEST(lambda cantonese: cantonese is None)
    )
    def flag_missing(self, req, mandarin):
        print(f"\n⚠️ SYSTEM 2: Cannot complete translation. Missing mapping for: {mandarin}")
        self.modify(req, processed=True)
        self.declare(MissingMapping(mandarin=mandarin))
        self.translation_done = False


class InteractiveTranslator:
    def __init__(self):
        self.engine = TranslationEngine()
        self.engine.reset()
        
        # Initial Collective Knowledgebase (Mandarin -> Cantonese)
        self.dictionary = {
            "你好": "你好 (nei5 hou2)",
            "什么": "咩 (me1)",
            "为什么": "點解 (dim2 gaai2)",
            "哪里": "邊度 (bin1 dou6)",
            "今天": "今日 (gam1 jat6)",
            "吃": "食 (sik6)",
            "没有": "冇 (mou5)",
            "看": "睇 (tai2)"
        }

    def chat_loop(self):
        print("==================================================")
        print("🤖 AGI Interactive Translator (Mandarin -> Cantonese)")
        print("Type a Mandarin word or short phrase. Type 'quit' to exit.")
        print("==================================================")
        
        while True:
            text = input("\nUser (Mandarin) > ").strip()
            if text.lower() == 'quit':
                break
                
            cantonese = self.dictionary.get(text, None)
            
            self.engine.translation_done = False
            self.engine.reset()
            self.engine.declare(TranslationRequest(
                mandarin=text,
                cantonese=cantonese,
                processed=False
            ))
            self.engine.run()
            
            if not self.engine.translation_done:
                print(f"AI > I don't know the Cantonese translation for '{text}'.")
                ans = input(f"AI > Can you teach me how to say '{text}' in Cantonese?\nUser (Cantonese) > ").strip()
                if ans:
                    self.dictionary[text] = ans
                    print(f"AI > Learned! Mandarin '{text}' -> Cantonese '{ans}'.")
                    print("AI > Re-running translation...")
                    
                    self.engine.reset()
                    self.engine.declare(TranslationRequest(
                        mandarin=text,
                        cantonese=ans,
                        processed=False
                    ))
                    self.engine.run()

if __name__ == "__main__":
    app = InteractiveTranslator()
    app.chat_loop()
