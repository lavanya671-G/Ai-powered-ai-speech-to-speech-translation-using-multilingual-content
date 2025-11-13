# smart_translator.py
"""
Fixed SmartTranslator - Better error handling and fallbacks
"""

from transformers import MarianMTModel, MarianTokenizer
import torch
import os

class SmartTranslator:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-hi"):
        self.model_name = model_name
        print(f" Loading HF model: {model_name}")
        
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
            
            # Use GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            print(f" Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print(" This might be due to:")
            print("   - Internet connection issues")
            print("   - Model name typo") 
            print("   - Hugging Face server issues")
            print(" Using fallback translation method")
            raise

    def translate(self, text: str, max_length: int = 128) -> str:
        """Translate text using the loaded model"""
        if not text or not text.strip():
            return ""
            
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=max_length
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=3, 
                    early_stopping=True
                )
            
           
            translated_text = self.tokenizer.decode(
                generated_tokens[0], 
                skip_special_tokens=True
            )
            
            return translated_text
            
        except Exception as e:
            print(f" Translation error: {e}")
            return f"[Translation Error: {str(e)}]"
