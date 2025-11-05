# translator.py

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import time
from db_json import save_line_translation

class EnhancedTranslator:
    HF_LANG_MAP = {
        'en': 'en', 'en-US': 'en', 'en-IN': 'en',
        'hi': 'hi', 'hi-IN': 'hi', 'hi-US': 'hi',
        'es': 'es', 'fr': 'fr', 'de': 'de',
        'it': 'it', 'ru': 'ru', 'ar': 'ar', 
        'zh': 'zh', 'nl': 'nl', 'pt': 'pt', 
        'ja': 'ja', 'ko': 'ko'
    }

    def __init__(self, model_name='facebook/m2m100_418M'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self.supported_languages = list(set(self.HF_LANG_MAP.keys()))  # All unique language codes

    def _normalize_lang_code(self, lang_code):
        """Normalize language code to M2M100 format"""
        return self.HF_LANG_MAP.get(lang_code, lang_code)

    def translate(self, text, source_lang, target_lang):
        """
        Translate a single string from source_lang to target_lang
        """
        try:
            # Normalize language codes
            src_norm = self._normalize_lang_code(source_lang)
            tgt_norm = self._normalize_lang_code(target_lang)
            
            # Skip if same language after normalization
            if src_norm == tgt_norm:
                return f"[Skipped: same language {src_norm}]"
            
            self.tokenizer.src_lang = src_norm
            
            encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)

            generated_tokens = self.model.generate(
                **encoded,
                forced_bos_token_id=self.tokenizer.get_lang_id(tgt_norm),
                num_beams=5,
                early_stopping=True,
                max_length=512
            )
            
            result = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

            # Save translation to DB
            try:
                save_line_translation(text, source_lang, {target_lang: result}, line_num=1, source_type="auto")
            except Exception as e:
                print(f"⚠️ Warning: failed to save translation to DB: {e}")

            return result
            
        except Exception as e:
            print(f"❌ Translation error ({source_lang}->{target_lang}): {e}")
            return f"[Error: {str(e)}]"

    def translate_to_all_languages(self, text, source_lang):
        """
        Translate the text to all supported languages except the source language
        """
        results = {}
        source_norm = self._normalize_lang_code(source_lang)
        
        for tgt in set(self.HF_LANG_MAP.values()):  # Use unique target languages
            if tgt == source_norm:
                continue  # Skip same language
            try:
                results[tgt] = self.translate(text, source_lang, tgt)
            except Exception as e:
                results[tgt] = f"[Error: {e}]"
        return results

    # -----------------------
    # Helper methods for Milestone-3
    # -----------------------
    def get_supported_languages(self):
        return list(set(self.HF_LANG_MAP.values()))  # Return unique languages

    def get_loaded_models(self):
        return [self.model_name]

    def batch_translate(self, text, source_lang, target_langs):
        """Batch translate to multiple languages efficiently"""
        try:
            source_norm = self._normalize_lang_code(source_lang)
            results = {}
            
            for target_lang in target_langs:
                if target_lang == source_norm:
                    results[target_lang] = f"[Skipped: same language {source_norm}]"
                    continue
                    
                try:
                    results[target_lang] = self.translate(text, source_lang, target_lang)
                except Exception as e:
                    results[target_lang] = f"[Error: {str(e)}]"
                    
            return results
        except Exception as e:
            print(f"❌ Batch translation error: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    translator = EnhancedTranslator()
    sample_text = "Hello, how are you?"
    print("Translating to all languages:")
    translations = translator.translate_to_all_languages(sample_text, "en")
    for lang, t in translations.items():
        print(f"{lang}: {t}")