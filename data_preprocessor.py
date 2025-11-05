# data_preprocessor.py

"""
ENHANCED DATA PREPROCESSOR
Step 4: Clean and preprocess data for training
"""

import re
import json
import os
import pandas as pd
from typing import Dict, List, Any

class EnhancedDataPreprocessor:
    def __init__(self):
        self.results_dir = "results"
        self.cleaned_data_dir = "results/cleaned_data"
        os.makedirs(self.cleaned_data_dir, exist_ok=True)
    # Add this method to the EnhancedDataPreprocessor class in data_preprocessor.py
    def prepare_training_data(self):
        """Step 4: Prepare clean training pairs - compatibility method"""
        print("\n🔄 PREPARING TRAINING DATA")
        print("=" * 40)
        
        # Load all data
        transcriptions = self.load_all_transcription_data()
        translations = self.load_all_translation_data()
        
        training_pairs = []
        
        # Create English-Hindi pairs from translations
        for pair in translations:
            if pair['source_lang'] == 'en' and pair['target_lang'] == 'hi':
                clean_source = self.clean_text(pair['source_text'])
                clean_target = self.clean_text(pair['target_text'])
                
                if clean_source and clean_target and len(clean_source) > 5 and len(clean_target) > 5:
                    training_pairs.append({
                        'en': clean_source,
                        'hi': clean_target,
                        'source': f"translation_{pair['source']}",
                        'quality': 'high'
                    })
        
        print(f"📊 Prepared {len(training_pairs)} training pairs")
        
        # Save the cleaned data
        self.save_cleaned_data(training_pairs)
        
        return training_pairs
    
    def load_all_transcription_data(self) -> Dict[str, List[Dict]]:
        """Load all transcription data from JSON files"""
        print("📂 Loading all transcription data...")
        
        all_data = {'en': [], 'hi': []}
        
        # Load from transcription files
        transcription_files = [
            "results/transcriptions/english_transcriptions.json",
            "results/transcriptions/hindi_transcriptions.json"
        ]
        
        for file_path in transcription_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    lang = 'en' if 'english' in file_path else 'hi'
                    
                    if isinstance(data, dict):
                        for key, entry in data.items():
                            if isinstance(entry, dict) and 'text' in entry:
                                all_data[lang].append({
                                    'text': entry['text'],
                                    'source': entry.get('source_type', 'unknown'),
                                    'timestamp': entry.get('timestamp', ''),
                                    'original_key': key
                                })
                    
                    print(f"✅ Loaded {len(all_data[lang])} {lang.upper()} transcriptions from {os.path.basename(file_path)}")
                    
                except Exception as e:
                    print(f"❌ Error loading {file_path}: {e}")
        
        return all_data
    
    def load_all_translation_data(self) -> List[Dict]:
        """Load all translation data for training pairs"""
        print("📂 Loading all translation data...")
        
        translation_pairs = []
        
        # Load from translation directories
        translation_dirs = [
            "results/translations/english_translations",
            "results/translations/hindi_translations"
        ]
        
        for dir_path in translation_dirs:
            if os.path.exists(dir_path):
                for file_name in os.listdir(dir_path):
                    if file_name.endswith('_translations.json'):
                        file_path = os.path.join(dir_path, file_name)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            if isinstance(data, dict):
                                for key, entry in data.items():
                                    if 'original_text' in entry and 'translated_text' in entry:
                                        source_lang = 'en' if 'english' in dir_path else 'hi'
                                        target_lang = file_name.replace('_translations.json', '')
                                        
                                        translation_pairs.append({
                                            'source_lang': source_lang,
                                            'target_lang': target_lang,
                                            'source_text': entry['original_text'],
                                            'target_text': entry['translated_text'],
                                            'source': entry.get('source_type', 'unknown'),
                                            'timestamp': entry.get('timestamp', '')
                                        })
                            
                        except Exception as e:
                            print(f"❌ Error loading {file_path}: {e}")
        
        print(f"✅ Loaded {len(translation_pairs)} translation pairs")
        return translation_pairs
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning for training data"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\(\)]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '').replace("'", "")
        
        # Remove multiple punctuation
        text = re.sub(r'[\.\,\!\?]{2,}', '.', text)
        
        return text
    
    def prepare_training_pairs(self) -> List[Dict]:
        """Step 4: Prepare clean training pairs"""
        print("\n🔄 PREPARING TRAINING DATA")
        print("=" * 40)
        
        # Load all data
        transcriptions = self.load_all_transcription_data()
        translations = self.load_all_translation_data()
        
        training_pairs = []
        
        # Create English-Hindi pairs from translations
        for pair in translations:
            if pair['source_lang'] == 'en' and pair['target_lang'] == 'hi':
                clean_source = self.clean_text(pair['source_text'])
                clean_target = self.clean_text(pair['target_text'])
                
                if clean_source and clean_target and len(clean_source) > 5 and len(clean_target) > 5:
                    training_pairs.append({
                        'en': clean_source,
                        'hi': clean_target,
                        'source': f"translation_{pair['source']}",
                        'quality': 'high'  # These are actual translations
                    })
        
        # Create synthetic pairs from transcriptions (if we have both English and Hindi)
        if len(transcriptions['en']) > 0 and len(transcriptions['hi']) > 0:
            # For demo, create some synthetic pairs (in real scenario, you'd need aligned data)
            min_samples = min(len(transcriptions['en']), len(transcriptions['hi']))
            for i in range(min(5, min_samples)):  # Limit to 5 synthetic pairs
                training_pairs.append({
                    'en': self.clean_text(transcriptions['en'][i]['text']),
                    'hi': f"[Synthetic Hindi translation of: {transcriptions['en'][i]['text'][:30]}...]",
                    'source': 'synthetic',
                    'quality': 'low'
                })
        
        print(f"📊 Prepared {len(training_pairs)} training pairs")
        print(f"   - High quality: {len([p for p in training_pairs if p.get('quality') == 'high'])}")
        print(f"   - Synthetic: {len([p for p in training_pairs if p.get('quality') == 'low'])}")
        
        return training_pairs
    
    def save_cleaned_data(self, training_pairs: List[Dict]):
        """Save cleaned training data"""
        print("\n💾 SAVING CLEANED TRAINING DATA")
        print("=" * 40)
        
        # Save as JSON
        json_file = os.path.join(self.cleaned_data_dir, "en_hi_training_pairs.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(training_pairs, f, ensure_ascii=False, indent=2)
        
        # Save as CSV for easy viewing
        csv_file = os.path.join(self.cleaned_data_dir, "en_hi_training_pairs.csv")
        df_data = []
        for pair in training_pairs:
            df_data.append({
                'english': pair['en'],
                'hindi': pair['hi'],
                'source': pair['source'],
                'quality': pair.get('quality', 'unknown')
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"✅ Cleaned data saved:")
        print(f"   📄 JSON: {json_file}")
        print(f"   📊 CSV: {csv_file}")
        
        return json_file, csv_file
    
    def get_data_statistics(self) -> Dict:
        """Get statistics about available data"""
        transcriptions = self.load_all_transcription_data()
        translations = self.load_all_translation_data()
        
        stats = {
            'total_english_transcriptions': len(transcriptions['en']),
            'total_hindi_transcriptions': len(transcriptions['hi']),
            'total_translation_pairs': len(translations),
            'english_hindi_pairs': len([p for p in translations if p['source_lang'] == 'en' and p['target_lang'] == 'hi']),
            'hindi_english_pairs': len([p for p in translations if p['source_lang'] == 'hi' and p['target_lang'] == 'en']),
            'other_translation_pairs': len([p for p in translations if p['source_lang'] not in ['en', 'hi'] or p['target_lang'] not in ['en', 'hi']])
        }
        
        return stats
# Add this alias at the end of data_preprocessor.py
class DataPreprocessor(EnhancedDataPreprocessor):
    """Backward compatibility alias"""
    pass
# Add backward compatibility alias
DataPreprocessor = EnhancedDataPreprocessor
