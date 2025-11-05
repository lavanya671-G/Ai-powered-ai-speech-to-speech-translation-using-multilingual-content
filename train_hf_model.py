# train_hf_model.py

"""
ENHANCED MODEL TRAINER
Step 5: Train model with JSON data + external datasets
"""

import os
import json
import time
import requests
import zipfile
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset, load_dataset
import torch

class EnhancedModelTrainer:
    def __init__(self):
        self.models_dir = "fine_tuned_models"
        self.external_data_dir = "external_data"
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.external_data_dir, exist_ok=True)
    
    def import_kaggle_datasets(self):
        """Step 5A: Import datasets from Kaggle or other sources"""
        print("\n🌐 IMPORTING EXTERNAL DATASETS")
        print("=" * 40)
        
        external_pairs = []
        
        # Method 1: Use Hugging Face datasets
        try:
            print("📥 Loading from Hugging Face datasets...")
            
            # Try to load a small parallel dataset
            try:
                # English-Hindi dataset example
                dataset = load_dataset("cfilt/iitb-english-hindi", split='train[:1000]')
                for item in dataset:
                    external_pairs.append({
                        'en': item['en'],
                        'hi': item['hi'],
                        'source': 'iitb_english_hindi'
                    })
                print(f"✅ Loaded {len(dataset)} samples from IITB English-Hindi")
            except:
                print("❌ IITB dataset not available")
            
        except Exception as e:
            print(f"❌ Hugging Face datasets error: {e}")
        
        # Method 2: Create synthetic sports commentary data
        sports_data = [
            {
                "en": "What an amazing goal by the striker!",
                "hi": "स्ट्राइकर द्वारा क्या शानदार गोल!",
                "source": "synthetic_sports"
            },
            {
                "en": "The team is playing excellent football today",
                "hi": "टीम आज शानदार फुटबॉल खेल रही है",
                "source": "synthetic_sports"
            },
            {
                "en": "Great save by the goalkeeper!",
                "hi": "गोलकीपर द्वारा शानदार सेव!",
                "source": "synthetic_sports"
            },
            {
                "en": "The match is getting very exciting",
                "hi": "मैच बहुत रोमांचक हो रहा है",
                "source": "synthetic_sports"
            },
            {
                "en": "That was a beautiful pass to the forward",
                "hi": "यह फॉरवर्ड को एक सुंदर पास था",
                "source": "synthetic_sports"
            },
            {
                "en": "The crowd is cheering loudly",
                "hi": "भीड़ जोर से जयकार कर रही है",
                "source": "synthetic_sports"
            },
            {
                "en": "What a fantastic match this is turning out to be",
                "hi": "यह मैच कितना शानदार हो रहा है",
                "source": "synthetic_sports"
            }
        ]
        
        external_pairs.extend(sports_data)
        print(f"✅ Added {len(sports_data)} synthetic sports commentary samples")
        
        # Save external data
        external_file = os.path.join(self.external_data_dir, "external_training_data.json")
        with open(external_file, 'w', encoding='utf-8') as f:
            json.dump(external_pairs, f, ensure_ascii=False, indent=2)
        
        print(f"📊 Total external samples: {len(external_pairs)}")
        return external_pairs
    
    def combine_all_training_data(self, json_pairs, external_pairs):
        """Combine JSON data with external datasets"""
        print("\n🔄 COMBINING ALL TRAINING DATA")
        print("=" * 40)
        
        all_pairs = []
        
        # Add JSON pairs (your collected data)
        for pair in json_pairs:
            if 'en' in pair and 'hi' in pair:
                all_pairs.append({
                    'en': pair['en'],
                    'hi': pair['hi'],
                    'source': pair.get('source', 'json_data')
                })
        
        # Add external pairs
        for pair in external_pairs:
            if 'en' in pair and 'hi' in pair:
                all_pairs.append({
                    'en': pair['en'],
                    'hi': pair['hi'],
                    'source': pair.get('source', 'external_data')
                })
        
        print(f"📊 Combined training data:")
        print(f"   - Your JSON data: {len(json_pairs)} pairs")
        print(f"   - External data: {len(external_pairs)} pairs")
        print(f"   - Total: {len(all_pairs)} pairs")
        
        # Remove duplicates based on English text
        unique_pairs = []
        seen_texts = set()
        
        for pair in all_pairs:
            text_hash = hash(pair['en'][:100])  # Hash first 100 chars to detect duplicates
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_pairs.append(pair)
        
        print(f"   - After deduplication: {len(unique_pairs)} unique pairs")
        
        return unique_pairs
    
    def fine_tune_translation_model(self, training_pairs, model_name="Helsinki-NLP/opus-mt-en-hi"):
        """Step 5B: Fine-tune translation model with combined data"""
        print(f"\n🎯 FINE-TUNING MODEL: {model_name}")
        print("=" * 50)
        
        if len(training_pairs) < 20:
            print(f"❌ Not enough training data. Have {len(training_pairs)} pairs, need at least 20.")
            return None
        
        print(f"📊 Training with {len(training_pairs)} samples")
        
        try:
            # Load tokenizer and model
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            
            # Prepare dataset in Hugging Face format
            dataset_dict = []
            for pair in training_pairs:
                dataset_dict.append({
                    "translation": {
                        "en": pair['en'],
                        "hi": pair['hi']
                    }
                })
            
            dataset = Dataset.from_list(dataset_dict)
            
            def preprocess_function(examples):
                inputs = [ex["en"] for ex in examples["translation"]]
                targets = [ex["hi"] for ex in examples["translation"]]
                
                model_inputs = tokenizer(
                    inputs, 
                    max_length=128, 
                    truncation=True, 
                    padding=True
                )
                
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(
                        targets, 
                        max_length=128, 
                        truncation=True, 
                        padding=True
                    )
                
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs
            
            # Tokenize dataset
            tokenized_dataset = dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            # Split into train/validation (80/20)
            train_size = int(0.8 * len(tokenized_dataset))
            train_dataset = tokenized_dataset.select(range(train_size))
            eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=os.path.join(self.models_dir, "fine_tuned_en_hi"),
                overwrite_output_dir=True,
                num_train_epochs=5,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir=os.path.join(self.models_dir, "logs"),
                logging_steps=50,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                report_to=None
            )
            
            # Data collator
            data_collator = DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                padding=True
            )
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer
            )
            
            # Start training
            print("🔄 Starting model training...")
            start_time = time.time()
            
            train_result = trainer.train()
            training_time = time.time() - start_time
            
            # Save metrics
            metrics = train_result.metrics
            trainer.save_metrics("train", metrics)
            
            print(f"✅ Training completed in {training_time:.2f} seconds")
            print(f"📈 Training loss: {metrics.get('train_loss', 'N/A')}")
            print(f"📊 Evaluation loss: {metrics.get('eval_loss', 'N/A')}")
            
            # Save the fine-tuned model
            model_save_path = os.path.join(self.models_dir, "fine_tuned_en_hi")
            trainer.save_model(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            
            print(f"💾 Model saved to: {model_save_path}")
            return model_save_path
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_if_necessary(self, json_training_pairs, min_samples=20):
        """Decide if training is necessary and train if needed"""
        print("\n🤔 CHECKING IF MODEL TRAINING IS NECESSARY")
        print("=" * 50)
        
        # Import external data
        external_pairs = self.import_kaggle_datasets()
        
        # Combine all data
        all_pairs = self.combine_all_training_data(json_training_pairs, external_pairs)
        
        if len(all_pairs) >= min_samples:
            print(f"✅ Sufficient data available ({len(all_pairs)} pairs). Starting training...")
            return self.fine_tune_translation_model(all_pairs)
        else:
            print(f"❌ Insufficient data. Have {len(all_pairs)} pairs, need {min_samples}.")
            print("💡 Using pre-trained models without fine-tuning.")
            return None
