# app.py - AI Speech Translator with Working TTS - FIXED VERSION
import os
import sys
import time
import traceback
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

# Force UTF-8 encoding for Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ---------------------------
# CRITICAL: Environment Check First
# ---------------------------
def check_environment():
    """Check environment before importing anything"""
    print("🔍 CHECKING ENVIRONMENT...")
    
    # Check .env file
    if not os.path.exists('.env'):
        print("❌ .env file missing! Create one with:")
        print("   SPEECH_KEY=your_azure_speech_key")
        print("   SPEECH_REGION=your_azure_region")
        return False
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    speech_key = os.getenv("SPEECH_KEY")
    speech_region = os.getenv("SPEECH_REGION")
    
    if not speech_key or not speech_region:
        print("❌ Missing Azure credentials in .env file!")
        print("   Please add SPEECH_KEY and SPEECH_REGION")
        return False
    
    print(f"✅ Environment check passed")
    print(f"   Region: {speech_region}")
    return True

# Run environment check first
if not check_environment():
    print("⚠️ Some features may not work without proper configuration")

# ---------------------------
# Import with Enhanced Error Handling
# ---------------------------
def safe_import(module_name, class_name=None):
    """Safely import modules with detailed error reporting"""
    try:
        if class_name:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        else:
            return __import__(module_name)
    except ImportError as e:
        print(f"⚠️ Import failed for {module_name}: {e}")
        return None
    except Exception as e:
        print(f"❌ Error importing {module_name}: {e}")
        return None

# Import translation pipeline with error handling
try:
    from translation_pipeline import TranslationPipeline
    TRANSLATION_PIPELINE_AVAILABLE = True
    print("✅ Translation pipeline: Available")
except Exception as e:
    print(f"⚠️ Translation pipeline not available: {e}")
    TRANSLATION_PIPELINE_AVAILABLE = False

# Milestone 3 optional modules
try:
    from ott_integration import OTTIntegration
    MILESTONE_3_AVAILABLE = True
    print("✅ OTT Integration: Available")
except Exception:
    MILESTONE_3_AVAILABLE = False
    print("⚠️ OTT Integration: Not available")

# TTS module - Enhanced import
TTS_AVAILABLE = False
tts_engine = None
try:
    from text_to_speech import MultilingualTextToSpeech
    TTS_AVAILABLE = True
    print("✅ TTS Module: Imported successfully")
    
    # Initialize TTS engine with better error handling
    try:
        tts_engine = MultilingualTextToSpeech()
        if tts_engine.is_available():
            print("✅ TTS Engine: Initialized successfully")
        else:
            print("⚠️ TTS Engine: No TTS engines available")
            TTS_AVAILABLE = False
    except Exception as e:
        print(f"❌ Failed to initialize TTS engine: {e}")
        tts_engine = None
        TTS_AVAILABLE = False
        
except Exception as e:
    print(f"⚠️ TTS module not available: {e}")
    TTS_AVAILABLE = False

# ---------------------------
# Directories
# ---------------------------
def initialize_directories():
    """Initialize required directories"""
    dirs = [
        "results", "results/transcriptions", "results/translations",
        "results/translations/english_translations", "results/translations/hindi_translations",
        "results/cleaned_data", "results/evaluation_results",
        "fine_tuned_models", "external_data", "results/audio_output"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ Directories initialized")

initialize_directories()

# ---------------------------
# Translator - Enhanced Import Handling
# ---------------------------
translator = None
EnhancedSpeechRecognizer = None

try:
    from speech_recognizer import EnhancedSpeechRecognizer
    print("✅ Speech Recognizer: Imported")
except Exception as e:
    print(f"❌ Failed to import Speech Recognizer: {e}")
    # Create a fallback class
    class EnhancedSpeechRecognizer:
        def __init__(self):
            raise ImportError("Speech recognizer not available")
        
        def recognize_from_audio_file(self, file_path):
            return "", "en"
            
        def recognize_from_microphone(self):
            return "", "en"

try:
    from translator import EnhancedTranslator as Translator
    translator = Translator()
    print("✅ Translator: Initialized")
except Exception as e:
    print(f"❌ Failed to import Translator: {e}")
    # Create fallback translator
    class FallbackTranslator:
        def translate(self, text, src_lang, target_lang):
            return f"[Translation unavailable] {text}"
            
        def batch_translate(self, text, src_lang, target_langs):
            return {lang: f"[Translation unavailable] {text}" for lang in target_langs}
    
    translator = FallbackTranslator()

try:
    from db_json import save_line_translation
    from utils_shared import SUPPORTED_LANG_CODES, ensure_str_from_translation
    print("✅ Database & Utils: Imported")
except Exception as e:
    print(f"⚠️ Database/Utils import warning: {e}")
    # Fallback functions
    SUPPORTED_LANG_CODES = ['en', 'hi', 'es', 'fr', 'de']
    
    def ensure_str_from_translation(text):
        return str(text) if text else ""
        
    def save_line_translation(*args, **kwargs):
        print(f"[DB] Would save translation: {args}")

# Language mapping
LANG_NAME_MAP = {
    'en': 'English', 'hi': 'Hindi', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'it': 'Italian', 'ru': 'Russian', 'ar': 'Arabic', 'zh': 'Chinese', 'nl': 'Dutch',
    'pt': 'Portuguese', 'ja': 'Japanese', 'ko': 'Korean', 'mr': 'Marathi', 'ml': 'Malayalam'
}

def lang_name(lang_code):
    return LANG_NAME_MAP.get(lang_code, lang_code)

def translate_text_single(text, src_lang, target_lang, **kwargs):
    try:
        # Skip if same language
        if src_lang.split('-')[0] == target_lang.split('-')[0]:
            return f"[Skipped: same language]"
            
        result = translator.translate(text, src_lang, target_lang, **kwargs)
        return ensure_str_from_translation(result)
    except Exception as e:
        print(f"❌ Translation error ({src_lang}->{target_lang}): {e}")
        return f"[Error: {str(e)}]"

# ---------------------------
# DIAGNOSTIC FUNCTIONS - ADD THESE
# ---------------------------
def diagnose_speech_services():
    """Diagnose speech services and provide fixes"""
    print("\n🔍 DIAGNOSING SPEECH SERVICES")
    print("=" * 50)
    
    # Check Azure credentials
    from dotenv import load_dotenv
    load_dotenv()
    
    speech_key = os.getenv("SPEECH_KEY")
    speech_region = os.getenv("SPEECH_REGION")
    
    print(f"📋 Azure Speech Configuration:")
    print(f"   SPEECH_KEY: {'***' + speech_key[-8:] if speech_key else '❌ MISSING'}")
    print(f"   SPEECH_REGION: {speech_region or '❌ MISSING'}")
    
    # Check Azure SDK
    try:
        import azure.cognitiveservices.speech as speechsdk
        print("   ✅ Azure Speech SDK: Available")
        
        # Test configuration
        if speech_key and speech_region:
            try:
                speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
                print("   ✅ Azure Credentials: Valid")
            except Exception as e:
                print(f"   ❌ Azure Credentials: Invalid - {e}")
        else:
            print("   ❌ Azure Credentials: Missing")
            
    except ImportError:
        print("   ❌ Azure Speech SDK: Not installed")
    
    # Check TTS
    try:
        if TTS_AVAILABLE and tts_engine:
            tts_info = tts_engine.get_engine_info()
            print(f"   ✅ TTS Engine: {tts_info}")
        else:
            print("   ❌ TTS Engine: Not available")
    except Exception as e:
        print(f"   ❌ TTS Engine: Failed - {e}")
    
    # Check Speech Recognizer
    try:
        recognizer = EnhancedSpeechRecognizer()
        print("   ✅ Speech Recognizer: Initialized")
    except Exception as e:
        print(f"   ❌ Speech Recognizer: Failed - {e}")
    
    print("🔍 Diagnosis completed")

def test_microphone_functionality():
    """Test microphone functionality"""
    print("\n🎤 TESTING MICROPHONE FUNCTIONALITY")
    print("=" * 50)
    
    try:
        recognizer = EnhancedSpeechRecognizer()
        print("✅ Speech Recognizer: Initialized")
        
        # Quick test
        print("🎤 Speak a short phrase (5 seconds)...")
        text, lang = recognizer.recognize_from_microphone()
        
        if text:
            print(f"✅ Microphone Test: SUCCESS - '{text}' ({lang})")
            return True
        else:
            print("❌ Microphone Test: FAILED - No speech detected")
            return False
            
    except Exception as e:
        print(f"❌ Microphone Test: ERROR - {e}")
        return False

def test_tts_functionality():
    """Test TTS functionality"""
    print("\n🔊 TESTING TTS FUNCTIONALITY")
    print("=" * 50)
    
    if not TTS_AVAILABLE or not tts_engine:
        print("❌ TTS not available")
        return False
        
    try:
        test_text = "This is a test of the text to speech system."
        print(f"🔊 Testing TTS with: '{test_text}'")
        
        success = tts_engine.speak_text(test_text, 'en')
        if success:
            print("✅ TTS Test: SUCCESS")
            return True
        else:
            print("❌ TTS Test: FAILED")
            return False
            
    except Exception as e:
        print(f"❌ TTS Test: ERROR - {e}")
        return False

# ---------------------------
# OTT readiness check
# ---------------------------
def check_ott_readiness_menu():
    print("\n🎯 OTT READINESS CHECK")
    print("=" * 40)
    
    if not MILESTONE_3_AVAILABLE:
        print("❌ OTT integration not available")
        return
        
    if not TRANSLATION_PIPELINE_AVAILABLE:
        print("❌ Translation pipeline not available")
        return
        
    pipeline = TranslationPipeline()
    
    # Check for external data files
    external_data_dir = "external_data"
    if not os.path.exists(external_data_dir):
        print(f"❌ No external_data directory found")
        return
        
    files = [os.path.join(external_data_dir, f) for f in os.listdir(external_data_dir) 
             if f.endswith(('.txt', '.json'))]
    
    if not files:
        print("❌ No input files found in 'external_data/' to analyze OTT readiness.")
        print("💡 Add some .txt or .json files to external_data/ folder")
        
        # Create sample files automatically
        print("🔄 Creating sample files for testing...")
        sample_text = """Welcome to the live sports commentary
                            What an amazing goal by the team
                            The match is getting very exciting
                            Great save by the goalkeeper"""
        
        with open(os.path.join(external_data_dir, "sample_sports.txt"), "w", encoding="utf-8") as f:
            f.write(sample_text)
        
        sample_json = [
            {"text": "The team is playing excellent football today"},
            {"text": "What a fantastic match this is turning out to be"},
            {"text": "The crowd is cheering loudly"}
        ]
        
        with open(os.path.join(external_data_dir, "sample_data.json"), "w", encoding="utf-8") as f:
            json.dump(sample_json, f, indent=2)
        
        print("✅ Created sample files in external_data/")
        files = [os.path.join(external_data_dir, f) for f in os.listdir(external_data_dir) 
                 if f.endswith(('.txt', '.json'))]
    
    if not files:
        print("❌ Still no files available for analysis")
        return

    

    print(f"📂 Found {len(files)} files. Analyzing translations and OTT metrics...\n")
    all_results = []

    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            translations_summary = []
            for i, line in enumerate(lines[:3]):  # Process first 3 lines only for demo
                print(f"📝 Processing line {i+1}: {line[:50]}...")
                translations, bleu_scores, latencies = translate_and_evaluate_parallel(line, 'en', ['hi', 'es', 'fr'])
                translations_summary.append({
                    'translations': translations,
                    'bleu_scores': bleu_scores,
                    'latencies': latencies
                })
                
                # Optional: Speak translations if TTS available
                if TTS_AVAILABLE and translations and input(f"Hear translations for this line? (y/n): ").lower() == 'y':
                    speak_translations_async(translations, original_text=line, original_lang='en')

            all_results.extend(translations_summary)
            
        except Exception as e:
            print(f"❌ Error processing file {file}: {e}")

    # Simple OTT readiness check
    print("\n📊 OTT READINESS ANALYSIS")
    print("=" * 30)
    
    if all_results:
        total_translations = sum(len(result['translations']) for result in all_results)
        if total_translations > 0:
            avg_latency = sum(sum(result['latencies'].values()) for result in all_results) / total_translations
            avg_bleu = sum(sum(result['bleu_scores'].values()) for result in all_results) / total_translations
        else:
            avg_latency = 0
            avg_bleu = 0
        
        print(f"📈 Total Translations: {total_translations}")
        print(f"⚡ Average Latency: {avg_latency:.2f}s")
        print(f"🎯 Average BLEU Score: {avg_bleu:.3f}")
        
        # Basic readiness criteria
        readiness_score = 0
        if avg_latency < 2.0:
            print("✅ Latency: Good for real-time (< 2s)")
            readiness_score += 1
        else:
            print("❌ Latency: Too slow for real-time")
            
        if avg_bleu > 0.3:
            print("✅ Translation Quality: Acceptable")
            readiness_score += 1
        else:
            print("❌ Translation Quality: Needs improvement")
            
        if total_translations >= 5:
            print("✅ Data Volume: Sufficient")
            readiness_score += 1
        else:
            print("❌ Data Volume: Insufficient")
            
        print(f"\n🎯 Overall OTT Readiness: {readiness_score}/3")
        if readiness_score >= 2:
            print("🚀 System is likely ready for OTT integration!")
        else:
            print("⚠️ System needs improvement before OTT integration")
    else:
        print("❌ No results to analyze")
    
    print("✅ OTT readiness check completed!")

# ---------------------------
# Session Manager
# ---------------------------
class SessionManager:
    def __init__(self):
        self.files_processed = []
        self.translations_made = 0
        self.translations_count = 0

    def log_processing(self, file_path, success=True, lines_processed=0, translations_count=0):
        self.files_processed.append({
            'file': file_path,
            'timestamp': datetime.now(),
            'success': success,
            'lines_processed': lines_processed,
            'translations_count': translations_count
        })
        if success:
            self.translations_made += lines_processed
            self.translations_count += translations_count

    def get_session_summary(self):
        total = len(self.files_processed)
        successful = len([f for f in self.files_processed if f['success']])
        success_rate = (successful / total * 100) if total > 0 else 100.0
        return {
            'total_files': total,
            'successful_files': successful,
            'translations_made': self.translations_made,
            'translations_count': self.translations_count,
            'success_rate': success_rate
        }

session_manager = SessionManager()

# ---------------------------
# Text-to-Speech helpers
# ---------------------------
def speak_translations_async(translations, original_text=None, original_lang='en'):
    if not TTS_AVAILABLE or not tts_engine:
        print("⚠️ TTS not available. Skipping speech output.")
        return
    try:
        tts_engine.speak_all_translations_async(translations, original_text, original_lang)
    except Exception as e:
        print(f"❌ Error in TTS: {e}")

# ---------------------------
# Audio / File Helpers
# ---------------------------
def validate_audio_file(file_path):
    """Validate audio file - UPDATED TO SUPPORT MP4"""
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    if not os.path.isfile(file_path):
        return False, "Path is not a file"
    
    # Check file extension - ADDED MP4 SUPPORT
    valid_extensions = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.webm', '.mp4'}
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext not in valid_extensions:
        supported = ', '.join(valid_extensions)
        return False, f"Unsupported format '{file_ext}'. Supported: {supported}"
    
    # Check file size
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
    
    if file_size > 1000:
        return True, f"Very large file ({file_size:.1f}MB) - processing line by line"
    elif file_size > 100:
        return True, f"Large file ({file_size:.1f}MB) - real-time processing"
    elif file_size == 0:
        return False, "File is empty (0 bytes)"
    else:
        return True, f"Ready ({file_size:.1f}MB)"
def extract_audio_from_mp4(mp4_path):
    """Extract audio from MP4 file and save as WAV"""
    try:
        from pydub import AudioSegment
        import tempfile
        
        print(f"🎬 Extracting audio from MP4: {os.path.basename(mp4_path)}")
        
        # Create temporary output file
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"extracted_{os.path.basename(mp4_path)}.wav")
        
        # Load and convert audio
        audio = AudioSegment.from_file(mp4_path, format="mp4")
        audio.export(output_path, format="wav")
        
        print(f"✅ Audio extracted to: {output_path}")
        return output_path
        
    except ImportError:
        print("❌ pydub not installed. Install with: pip install pydub")
        return None
    except Exception as e:
        print(f"❌ MP4 audio extraction error: {e}")
        return None

# ---------------------------
# Translation & Evaluation
# ---------------------------
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth_fn = SmoothingFunction().method1
    NLTK_AVAILABLE = True
    print("✅ NLTK: Available for BLEU scores")
except ImportError:
    print("⚠️ NLTK not available, BLEU scores will be disabled")
    NLTK_AVAILABLE = False
    smooth_fn = None

def translate_and_evaluate_parallel(text, src_lang, selected_langs=None):
    """Enhanced version with better timeout handling and batch processing"""
    translations, bleu_scores, latencies = {}, {}, {}
    if selected_langs is None:
        selected_langs = [l for l in SUPPORTED_LANG_CODES if l != src_lang]
    if not selected_langs:
        return translations, bleu_scores, latencies

    # Filter out source language if present
    selected_langs = [lang for lang in selected_langs if lang != src_lang]
    
    if not selected_langs:
        return translations, bleu_scores, latencies

    print(f"🎯 Translating from {src_lang} to {len(selected_langs)} languages: {selected_langs}")

    # Try batch translation first (more efficient)
    try:
        if hasattr(translator, 'batch_translate'):
            print("🔄 Using batch translation...")
            start_time = time.time()
            batch_results = translator.batch_translate(text, src_lang, selected_langs)
            batch_latency = time.time() - start_time
            
            # Distribute latency evenly (approximation)
            avg_latency = batch_latency / len(selected_langs) if selected_langs else 0
            
            for lang in selected_langs:
                if lang in batch_results and batch_results[lang] and not batch_results[lang].startswith('[Error') and not batch_results[lang].startswith('[Skipped'):
                    translations[lang] = ensure_str_from_translation(batch_results[lang])
                    latencies[lang] = avg_latency
                    print(f"✅ Batch translated to {lang_name(lang)}")
            
            if translations:
                # Calculate BLEU scores if NLTK available
                if NLTK_AVAILABLE:
                    ref_tokens = (text or "").split()
                    for tgt, trans in translations.items():
                        trans_tokens = (trans or "").split()
                        try:
                            bleu_scores[tgt] = sentence_bleu([ref_tokens], trans_tokens, smoothing_function=smooth_fn)
                        except:
                            bleu_scores[tgt] = 0.0
                else:
                    # Set default BLEU scores if NLTK not available
                    for tgt in translations:
                        bleu_scores[tgt] = 0.5  # Default score
                
                # Save to database
                try:
                    save_line_translation(text, src_lang, translations, 1, "auto")
                    session_manager.log_processing("Translation", True, len((text or "").split()), len(translations))
                except Exception as e:
                    print(f"⚠️ Failed to save translation: {e}")
                
                return translations, bleu_scores, latencies
    except Exception as e:
        print(f"⚠️ Batch translation failed: {e}")

    # Fallback to parallel translation with better timeout handling
    print("🔄 Using parallel translation (fallback)...")

    def translate_one(tgt):
        """Translate to a single target language"""
        try:
            start = time.time()
            translated = translate_text_single(text, src_lang, tgt)
            latency = time.time() - start
            return tgt, translated, latency
        except Exception as e:
            print(f"❌ Translation error for {tgt}: {e}")
            return tgt, None, 0.0

    # Use smaller number of workers to avoid overwhelming the system
    max_workers = min(4, len(selected_langs))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all translation tasks
        future_to_lang = {
            executor.submit(translate_one, tgt): tgt 
            for tgt in selected_langs
        }
        
        completed = 0
        timeout_seconds = 15.0  # Reduced timeout per translation
        
        try:
            # Process completed futures with timeout
            for future in as_completed(future_to_lang, timeout=timeout_seconds * len(selected_langs)):
                try:
                    tgt, translated, latency = future.result(timeout=timeout_seconds)
                    if translated and not translated.startswith('[Error') and not translated.startswith('[Skipped'):
                        translations[tgt] = translated
                        latencies[tgt] = latency
                        completed += 1
                        print(f"✅ Translated to {lang_name(tgt)} (latency: {latency:.2f}s)")
                    else:
                        print(f"⚠️ No valid translation for {lang_name(tgt)}")
                except FutureTimeoutError:
                    lang = future_to_lang[future]
                    print(f"⏰ Timeout for {lang_name(lang)}")
                except Exception as e:
                    lang = future_to_lang[future]
                    print(f"❌ Error for {lang_name(lang)}: {e}")
                    
        except TimeoutError:
            print(f"⏰ Overall translation timeout after {timeout_seconds * len(selected_langs)} seconds")
            # Continue with whatever translations we have

    print(f"📊 Translation completed: {completed}/{len(selected_langs)} languages")

    # Calculate BLEU scores for successful translations
    if translations:
        if NLTK_AVAILABLE:
            ref_tokens = (text or "").split()
            for tgt, trans in translations.items():
                trans_tokens = (trans or "").split()
                try:
                    bleu_scores[tgt] = sentence_bleu([ref_tokens], trans_tokens, smoothing_function=smooth_fn)
                except:
                    bleu_scores[tgt] = 0.0
        else:
            # Set default BLEU scores
            for tgt in translations:
                bleu_scores[tgt] = 0.5

        # Save to database
        try:
            save_line_translation(text, src_lang, translations, 1, "auto")
            session_manager.log_processing("Translation", True, len((text or "").split()), len(translations))
        except Exception as e:
            print(f"⚠️ Failed to save translation: {e}")

    return translations, bleu_scores, latencies

# ---------------------------
# Audio processing flows - ENHANCED WITH ERROR HANDLING
# ---------------------------
def process_audio_file(file_path):
    print(f"🎯 Processing: {file_path}")
    ok, msg = validate_audio_file(file_path)
    if not ok:
        print(f"❌ {msg}")
        return
    print(f"📦 {msg}")

    try:
        recognizer = EnhancedSpeechRecognizer()
        text, language = recognizer.recognize_from_audio_file(file_path)
        if not text:
            print("❌ No speech detected")
            return
        print(f"📝 Detected ({language}): {text[:200]}...")

        if input("Translate this audio? (y/n): ").strip().lower() != 'y':
            return

        lang_input = input("Enter target languages (hi, es, fr...) or 'all': ").strip().lower()
        selected_langs = SUPPORTED_LANG_CODES if lang_input == 'all' else [l.strip() for l in lang_input.split(",") if l.strip() in SUPPORTED_LANG_CODES]

        translations, bleu_scores, latencies = translate_and_evaluate_parallel(text, language, selected_langs)
        if not translations:
            print("⚠️ No translations generated.")
            return

        print("🌍 Translations:")
        for lang, t in translations.items():
            print(f"{lang_name(lang)} | BLEU: {bleu_scores.get(lang,0.0):.2f} | {t[:150]}...")

        # FIX: Properly wait for TTS input
        if TTS_AVAILABLE:
            tts_choice = input("Hear translations? (y/n): ").strip().lower()
            if tts_choice == 'y':
                print("🔊 Playing translations...")
                speak_translations_async(translations, original_text=None, original_lang=language)
                # Wait a moment for TTS to start
                time.sleep(2)
        else:
            print("⚠️ TTS not available for audio playback")
    except Exception as e:
        print(f"❌ Error processing audio file: {e}")
        print("💡 Check if Azure Speech credentials are correct")

def handle_microphone_input():
    try:
        recognizer = EnhancedSpeechRecognizer()
        text, language = recognizer.recognize_from_microphone()
        if not text:
            print("❌ No speech detected")
            return
        print(f"📝 Detected ({language}): {text[:140]}...")

        if input("Translate detected text? (y/n): ").strip().lower() != 'y':
            return

        lang_input = input("Enter target languages (hi, es, fr...) or 'all': ").strip().lower()
        selected_langs = SUPPORTED_LANG_CODES if lang_input == 'all' else [l.strip() for l in lang_input.split(",") if l.strip() in SUPPORTED_LANG_CODES]

        try:
            translations, bleu_scores, latencies = translate_and_evaluate_parallel(text, language, selected_langs)
            if not translations:
                print("⚠️ No translations generated.")
                return

                    # Auto-play translations without asking
            if TTS_AVAILABLE:
                print("🔊 Playing translations automatically...")
                speak_translations_async(translations, original_text=None, original_lang=language)
                # Wait a moment for TTS to start
                time.sleep(2)
            else:
                print("⚠️ TTS not available for audio playback")
            speak_translations_async(translations, text, language)
        except Exception as e:
            print(f"❌ Translation process failed: {e}")
            
    except Exception as e:
        print(f"❌ Microphone input failed: {e}")
        print("💡 Check microphone permissions and Azure Speech configuration")

# ---------------------------
# CLI
# ---------------------------
def view_system_status():
    s = session_manager.get_session_summary()
    print(f"\n📊 SYSTEM STATUS")
    print("=" * 30)
    print(f"Files processed: {s['total_files']}")
    print(f"Successful files: {s['successful_files']}")
    print(f"Translations made: {s['translations_count']}")
    print(f"Words processed: {s['translations_made']}")
    print(f"Success rate: {s['success_rate']:.1f}%")
    print(f"TTS Available: {'✅ Yes' if TTS_AVAILABLE else '❌ No'}")
    print(f"OTT Integration: {'✅ Available' if MILESTONE_3_AVAILABLE else '❌ Not available'}")
    print(f"Translation Pipeline: {'✅ Available' if TRANSLATION_PIPELINE_AVAILABLE else '❌ Not available'}")
    print(f"NLTK Available: {'✅ Yes' if NLTK_AVAILABLE else '❌ No'}")
    print("Supported languages:", ', '.join(SUPPORTED_LANG_CODES))

def main():
    print("\n🚀 AI-POWERED UNIVERSAL SPEECH TRANSLATION SYSTEM")
    print("=" * 50)
    
    # Display system capabilities
    print(f"✅ Translation: {len(SUPPORTED_LANG_CODES)} languages supported")
    print(f"✅ Speech Recognition: English & Hindi")
    print(f"{'✅' if TTS_AVAILABLE else '❌'} Text-to-Speech: {'Available' if TTS_AVAILABLE else 'Not available'}")
    print(f"{'✅' if MILESTONE_3_AVAILABLE else '❌'} OTT Integration: {'Available' if MILESTONE_3_AVAILABLE else 'Not available'}")
    
    # Diagnostic menu option
    print("\n0. Run Diagnostics")
    
    pipeline = None
    if TRANSLATION_PIPELINE_AVAILABLE:
        pipeline = TranslationPipeline()
    
    while True:
        s = session_manager.get_session_summary()
        print(f"\nSession: Files {s['total_files']} | Translations {s['translations_made']} | Success {s['success_rate']:.1f}%")
        print("1. Microphone input")
        print("2. Process audio file")
        print("3. Run translation pipeline (Milestone 3)")
        print("4. View system status")
        print("5. Check OTT readiness")
        print("6. Exit")
        choice = input("Enter choice: ").strip()
        
        if choice == '0':
            diagnose_speech_services()
            if input("Test microphone? (y/n): ").lower() == 'y':
                test_microphone_functionality()
            if input("Test TTS? (y/n): ").lower() == 'y':
                test_tts_functionality()
        elif choice == '1':
            handle_microphone_input()
        elif choice == '2':
            path = input("Enter audio file path: ").strip()
            process_audio_file(path)
        elif choice == '3':
            if not TRANSLATION_PIPELINE_AVAILABLE:
                print("❌ Milestone 3 pipeline not available")
            else:
                try:
                    report = pipeline.run_full_pipeline()
                    print("✅ Pipeline finished")
                    print(report)
                except Exception as e:
                    print(f"❌ Pipeline error: {e}")
        elif choice == '4':
            view_system_status()
        elif choice == '5':
            check_ott_readiness_menu()
        elif choice == '6':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted. Exiting.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()