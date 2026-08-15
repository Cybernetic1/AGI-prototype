import sys
import collections
import collections.abc
collections.Mapping = collections.abc.Mapping
from experta import *
import jieba

class TranslationRequest(Fact): pass
class ValidTranslation(Fact): pass
class MissingMapping(Fact): pass

class TranslationEngine(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self.translation_done = False
        
    @Rule(
        AS.req << TranslationRequest(
            mandarin_tokens=MATCH.mandarin_tokens,
            cantonese_tokens=MATCH.cantonese_tokens,
            processed=False
        ),
        TEST(lambda cantonese_tokens: all(t is not None for t in cantonese_tokens))
    )
    def output_translation(self, req, mandarin_tokens, cantonese_tokens):
        print(f"\n✅ SYSTEM 2 (Translation Engine): Valid mapping found.")
        print(f"  -> Mandarin Input Tokens: {mandarin_tokens}")
        
        # Simple reconstruction (Cantonese usually doesn't need spaces, but keeping them for clarity here)
        out_str = "".join([t.split(" (")[0] for t in cantonese_tokens])
        
        print(f"  -> Cantonese Output: {out_str}\n")
        self.modify(req, processed=True)
        self.declare(ValidTranslation(mandarin_tokens=mandarin_tokens, cantonese_tokens=cantonese_tokens))
        self.translation_done = True

    @Rule(
        AS.req << TranslationRequest(
            mandarin_tokens=MATCH.mandarin_tokens,
            cantonese_tokens=MATCH.cantonese_tokens,
            processed=False
        ),
        TEST(lambda cantonese_tokens: any(t is None for t in cantonese_tokens))
    )
    def flag_missing(self, req, mandarin_tokens, cantonese_tokens):
        # Find the first missing token
        missing_idx = cantonese_tokens.index(None)
        missing_word = mandarin_tokens[missing_idx]
        
        print(f"\n⚠️ SYSTEM 2: Cannot complete translation. Missing mapping for token: '{missing_word}'")
        self.modify(req, processed=True)
        self.declare(MissingMapping(mandarin=missing_word))
        self.translation_done = False


class InteractiveJiebaTranslator:
    def __init__(self):
        self.engine = TranslationEngine()
        self.engine.reset()
        
        # Initial Collective Knowledgebase (Mandarin -> Cantonese)
        self.dictionary = {
            "你": "你 (nei5)",
            "好": "好 (hou2)",
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
        print("🤖 AGI Interactive Translator (Jieba Tokenized)")
        print("Type a Mandarin sentence. Type 'quit' to exit.")
        print("==================================================")
        
        while True:
            text = input("\nUser (Mandarin) > ").strip()
            if text.lower() == 'quit':
                break
                
            # System 1: Tokenize using Jieba
            tokens = list(jieba.cut(text))
            
            # Map tokens
            cantonese_tokens = [self.dictionary.get(t, None) for t in tokens]
            
            self.engine.translation_done = False
            self.engine.reset()
            self.engine.declare(TranslationRequest(
                mandarin_tokens=tokens,
                cantonese_tokens=cantonese_tokens,
                processed=False
            ))
            self.engine.run()
            
            if not self.engine.translation_done:
                # Engine will have flagged the exact missing token via MissingMapping fact
                # Find it in the working memory
                missing = None
                for f in self.engine.facts.values():
                    if type(f).__name__ == "MissingMapping":
                        missing = f.get("mandarin")
                        break
                        
                if missing:
                    ans = input(f"AI > Can you teach me how to say '{missing}' in Cantonese?\nUser (Cantonese) > ").strip()
                    if ans:
                        self.dictionary[missing] = ans
                        print(f"AI > Learned! Mandarin '{missing}' -> Cantonese '{ans}'.")
                        print("AI > Re-running translation...")
                        
                        # Re-map with new dictionary
                        cantonese_tokens = [self.dictionary.get(t, None) for t in tokens]
                        
                        self.engine.reset()
                        self.engine.declare(TranslationRequest(
                            mandarin_tokens=tokens,
                            cantonese_tokens=cantonese_tokens,
                            processed=False
                        ))
                        self.engine.run()

if __name__ == "__main__":
    app = InteractiveJiebaTranslator()
    app.chat_loop()
