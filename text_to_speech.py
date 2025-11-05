"""
OPTIMIZED MULTILINGUAL TEXT-TO-SPEECH MODULE
- Fixed Azure TTS result checking
- Optimized for 12 core languages
- Better error handling
"""

import os
import time
import threading
import tempfile
import re
from datetime import datetime

# Try to import pyttsx3 for offline TTS
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("⚠️ pyttsx3 not available - offline TTS disabled")

# Try to import Azure Cognitive Services for cloud TTS
try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("⚠️ Azure Speech SDK not available - cloud TTS disabled")

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not available - using system environment variables")

class MultilingualTextToSpeech:
    """
    Optimized Text-to-Speech engine for 12 core languages
    """
    
    # Optimized voice mapping for 12 core languages
    AZURE_VOICE_MAPPING = {
        'en': 'en-US-JennyNeural',
        'hi': 'hi-IN-MadhurNeural', 
        'es': 'es-ES-ElviraNeural',
        'fr': 'fr-FR-DeniseNeural',
        'de': 'de-DE-KatjaNeural',
        'it': 'it-IT-ElsaNeural',
        'ru': 'ru-RU-SvetlanaNeural',
        'ar': 'ar-EG-SalmaNeural',
        'zh': 'zh-CN-XiaoxiaoNeural',
        'nl': 'nl-NL-ColetteNeural',
        'pt': 'pt-BR-FranciscaNeural',
        'ja': 'ja-JP-NanamiNeural',
        'ko': 'ko-KR-SunHiNeural'
    }
    
    # Language name mapping for display
    LANGUAGE_NAMES = {
        'en': 'English', 
        'hi': 'Hindi', 
        'es': 'Spanish', 
        'fr': 'French',
        'de': 'German', 
        'it': 'Italian', 
        'ru': 'Russian', 
        'ar': 'Arabic',
        'zh': 'Chinese', 
        'nl': 'Dutch', 
        'pt': 'Portuguese', 
        'ja': 'Japanese',
        'ko': 'Korean'
    }

    def __init__(self):
        """Initialize the TTS engine with available providers"""
        print("🔊 INITIALIZING MULTILINGUAL TEXT-TO-SPEECH ENGINE")
        print("=" * 50)
        
        # Create output directory
        self.output_dir = "results/audio_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Thread safety
        self.is_speaking = False
        self._lock = threading.Lock()
        
        # Initialize Azure TTS if available
        self.azure_enabled = False
        self.azure_speech_config = None
        
        # Get Azure credentials
        self.speech_key = os.getenv("SPEECH_KEY")
        self.speech_region = os.getenv("SPEECH_REGION")
        
        if AZURE_AVAILABLE and self.speech_key and self.speech_region:
            try:
                self.azure_speech_config = speechsdk.SpeechConfig(
                    subscription=self.speech_key, 
                    region=self.speech_region
                )
                self.azure_enabled = True
                print(f"✅ Azure TTS enabled - Region: {self.speech_region}")
                print(f"🎯 Supporting {len(self.AZURE_VOICE_MAPPING)} languages")
            except Exception as e:
                print(f"❌ Azure TTS configuration failed: {e}")
                self.azure_enabled = False
        else:
            if not self.speech_key or not self.speech_region:
                print("⚠️ Azure TTS disabled - Missing SPEECH_KEY or SPEECH_REGION in .env")
            else:
                print("⚠️ Azure TTS disabled - Azure SDK not available")
        
        # Initialize pyttsx3 offline TTS
        self.pyttsx3_available = False
        self.pyttsx3_engine = None
        
        if PYTTSX3_AVAILABLE:
            try:
                self.pyttsx3_engine = pyttsx3.init()
                # Configure pyttsx3 for better quality
                voices = self.pyttsx3_engine.getProperty('voices')
                self.pyttsx3_engine.setProperty('rate', 180)
                self.pyttsx3_engine.setProperty('volume', 0.9)
                if len(voices) > 1:
                    self.pyttsx3_engine.setProperty('voice', voices[1].id)
                self.pyttsx3_available = True
                print("✅ pyttsx3 offline TTS engine initialized")
                print("⚠️ Note: pyttsx3 has limited language support")
            except Exception as e:
                print(f"❌ Failed to initialize pyttsx3: {e}")
                self.pyttsx3_available = False
        else:
            print("⚠️ pyttsx3 not available - offline TTS disabled")
        
        print(f"📊 TTS Status: Azure={self.azure_enabled}, pyttsx3={self.pyttsx3_available}")

    def text_to_speech(self, text, language_code='en', output_filename=None):
        """
        Convert text to speech using available TTS engines
        """
        if not text or not str(text).strip():
            print("⚠️ No text to speak")
            return False
        
        # Try Azure TTS first if enabled
        if self.azure_enabled:
            azure_success = self._speak_with_azure(text, language_code, output_filename)
            if azure_success:
                return True
            else:
                print("🔄 Azure TTS failed, trying pyttsx3 fallback...")
        
        # Fallback to pyttsx3
        if self.pyttsx3_available:
            return self._speak_with_pyttsx3(text, language_code, output_filename)
        else:
            print("❌ No TTS engine available")
            return False

    def _normalize_language_code(self, language_code):
        """Normalize language code to standard format"""
        if not language_code:
            return 'en'
        lang = str(language_code).lower().strip()
        if '-' in lang:
            return lang.split('-')[0]
        return lang

    def _normalize_text(self, text):
        """Ensure text is in proper string format"""
        if text is None:
            return ""
        if isinstance(text, list):
            return ' '.join(str(item) for item in text)
        if isinstance(text, dict):
            return str(text.get('translation_text', text))
        return str(text)

    def _split_into_sentences(self, text):
        """Split text into sentences for better TTS quality"""
        # Split by common sentence endings
        sentences = re.split(r'[.!?।！？]+', text)
        # Filter out empty strings and add punctuation back
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        return sentences

    def _speak_with_azure(self, text, language_code, output_filename=None):
        """Use Azure Cognitive Services for TTS - OPTIMIZED VERSION"""
        try:
            # Normalize inputs
            clean_text = self._normalize_text(text).strip()
            if not clean_text:
                print("⚠️ Empty text provided to Azure TTS")
                return False
                
            # Add punctuation if missing
            if not clean_text.endswith(('.', '!', '?', '।', '！', '？')):
                clean_text += '.'
                
            lang_code = self._normalize_language_code(language_code)
            voice_name = self.AZURE_VOICE_MAPPING.get(lang_code, 'en-US-JennyNeural')
            language_name = self.LANGUAGE_NAMES.get(lang_code, lang_code)
            
            print(f"🔊 Azure TTS: {language_name} - '{clean_text[:60]}...'")
            
            # Create fresh speech config for each request
            speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key, 
                region=self.speech_region
            )
            speech_config.speech_synthesis_voice_name = voice_name
            
            # Use high quality audio format
            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
            )
            
            # Setup audio output
            if output_filename:
                output_path = os.path.join(self.output_dir, output_filename)
                audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
                print(f"💾 Saving to: {output_filename}")
            else:
                audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
            
            # Create synthesizer
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=audio_config
            )
            
            # Process text - break long texts into chunks
            if len(clean_text) > 100:
                print("📝 Long text detected, breaking into chunks...")
                sentences = self._split_into_sentences(clean_text)
                all_success = True
                
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        print(f"🔊 Speaking chunk {i+1}/{len(sentences)}: {sentence[:50]}...")
                        result = synthesizer.speak_text_async(sentence).get()
                        
                        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                            all_success = False
                            print(f"❌ Chunk {i+1} failed")
                        
                        # Small pause between chunks
                        time.sleep(0.3)
                
                return all_success
            else:
                # Short text - process normally
                result = synthesizer.speak_text_async(clean_text).get()
                
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    print(f"✅ Azure TTS completed successfully!")
                    if output_filename and os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        print(f"💾 Audio file saved: {output_filename} ({file_size} bytes)")
                    return True
                elif result.reason == speechsdk.ResultReason.Canceled:
                    cancellation_details = result.cancellation_details
                    print(f"❌ Azure TTS canceled: {cancellation_details.reason}")
                    if cancellation_details.error_details:
                        print(f"   Error details: {cancellation_details.error_details}")
                    return False
                else:
                    print(f"⚠️ Azure TTS returned: {result.reason}")
                    return False
                
        except Exception as e:
            print(f"❌ Azure TTS exception: {e}")
            return False

    def _speak_with_pyttsx3(self, text, language_code=None, output_filename=None):
        """Use pyttsx3 for offline TTS"""
        try:
            clean_text = self._normalize_text(text)
            if not clean_text or not clean_text.strip():
                print("⚠️ Empty text provided to pyttsx3")
                return False
                
            if not self.pyttsx3_engine:
                print("❌ pyttsx3 engine not available")
                return False
            
            lang_name = self.LANGUAGE_NAMES.get(
                self._normalize_language_code(language_code), 
                "Unknown"
            )
            
            print(f"🔊 pyttsx3 TTS: {lang_name} - '{clean_text[:60]}...'")
            
            if output_filename:
                print(f"⚠️ pyttsx3 file output limited - playing audio instead")
                output_path = os.path.join(self.output_dir, output_filename)
                try:
                    with open(output_path, 'w') as f:
                        f.write(f"TTS Audio placeholder for: {clean_text}")
                    print(f"📄 Created placeholder file: {output_filename}")
                except Exception as file_error:
                    print(f"⚠️ Could not create placeholder file: {file_error}")
            
            # Speak the text
            self.pyttsx3_engine.say(clean_text)
            self.pyttsx3_engine.runAndWait()
            print("✅ pyttsx3 speech completed")
            return True
            
        except Exception as e:
            print(f"❌ pyttsx3 TTS error: {e}")
            return False

    def synthesize_to_file(self, text, language_code, output_filename):
        """
        Synthesize speech directly to audio file
        """
        print(f"💾 Synthesizing to file: {output_filename}")
        print(f"📝 Text: {text[:80]}...")
        print(f"🌐 Language: {language_code}")
        
        return self.text_to_speech(text, language_code, output_filename)

    def speak_text(self, text, language_code='en'):
        """
        Speak text with thread safety
        """
        with self._lock:
            if self.is_speaking:
                print("⚠️ TTS is already speaking, waiting...")
                time.sleep(2.0)
            
            self.is_speaking = True
            try:
                # Small delay before starting
                time.sleep(0.3)
                
                # Use Azure TTS for better quality
                if self.azure_enabled:
                    return self._speak_with_azure(text, language_code)
                else:
                    print("❌ Azure TTS not available")
                    return False
                    
            except Exception as e:
                print(f"❌ Error in speak_text: {e}")
                return False
            finally:
                self.is_speaking = False
                # Cleanup delay
                time.sleep(0.3)

    def speak_all_translations(self, translations, original_text=None, original_lang='en'):
        """
        Speak all translations in sequence
        """
        print(f"\n🎵 Speaking {len(translations)} translations")
        print("=" * 40)
        
        if original_text:
            print(f"📝 Original ({original_lang}): {original_text[:80]}...")
        
        success_count = 0
        total_attempted = 0
        
        for lang_code, translated_text in translations.items():
            if (translated_text and 
                str(translated_text).strip() and 
                not str(translated_text).startswith(('[Error', '[Skipped'))):
                
                lang_norm = self._normalize_language_code(lang_code)
                lang_name = self.LANGUAGE_NAMES.get(lang_norm, lang_norm)
                total_attempted += 1
                
                print(f"\n🔊 [{total_attempted}/{len(translations)}] {lang_name}: {translated_text[:60]}...")
                
                try:
                    # Delay before speaking
                    time.sleep(1.5)
                    
                    if self.speak_text(translated_text, lang_norm):
                        success_count += 1
                        print(f"✅ Successfully spoke {lang_name}")
                    else:
                        print(f"❌ Failed to speak {lang_name}")
                    
                    # Pause after each language
                    time.sleep(2.0)
                    
                except Exception as e:
                    print(f"❌ Error speaking {lang_name}: {e}")
        
        print(f"\n📊 TTS Summary: {success_count}/{total_attempted} successful")
        return success_count > 0

    def speak_all_translations_async(self, translations, original_text=None, original_lang='en'):
        """
        Speak translations in a separate thread
        """
        thread = threading.Thread(
            target=self.speak_all_translations,
            args=(translations.copy(), original_text, original_lang),
            daemon=True
        )
        thread.start()
        return thread

    def is_available(self):
        """Check if any TTS engine is available"""
        return self.azure_enabled or self.pyttsx3_available

    def get_engine_info(self):
        """Get information about available TTS engines"""
        return {
            'azure_enabled': self.azure_enabled,
            'pyttsx3_available': self.pyttsx3_available,
            'speech_region': self.speech_region if self.azure_enabled else None,
            'available_languages': list(self.LANGUAGE_NAMES.keys()),
            'output_directory': self.output_dir,
            'total_languages': len(self.LANGUAGE_NAMES)
        }

    def test_single_language(self, language_code='en', test_text=None):
        """Test a single language with custom text"""
        if not test_text:
            test_texts = {
                'en': 'Hello, this is a clear voice test.',
                'hi': 'नमस्ते, यह स्पष्ट आवाज परीक्षण है।',
                'es': 'Hola, esta es una prueba de voz clara.',
                'fr': 'Bonjour, ceci est un test vocal clair.',
            }
            test_text = test_texts.get(language_code, 'Test message.')
        
        print(f"\n🧪 Testing {self.LANGUAGE_NAMES.get(language_code, language_code)}")
        print(f"📝 Text: {test_text}")
        
        success = self.speak_text(test_text, language_code)
        print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        return success

    def test_all_languages(self):
        """Test all 12 core languages"""
        print("\n🧪 TESTING ALL LANGUAGES")
        print("=" * 30)
        
        test_texts = {
            'en': 'Hello, this is a voice test in English.',
            'hi': 'नमस्ते, यह हिंदी में आवाज परीक्षण है।',
            'es': 'Hola, esta es una prueba de voz en español.',
            'fr': 'Bonjour, ceci est un test vocal en français.',
            'de': 'Hallo, dies ist ein Sprachtest auf Deutsch.',
            'it': 'Ciao, questo è un test vocale in italiano.',
            'ru': 'Привет, это голосовой тест на русском.',
            'ar': 'مرحبا، هذا اختبار صوتي باللغة العربية.',
            'zh': '你好，这是中文语音测试。',
            'nl': 'Hallo, dit is een stemtest in het Nederlands.',
            'pt': 'Olá, este é um teste de voz em português.',
            'ja': 'こんにちは、これは日本語での音声テストです。',
            'ko': '안녕하세요, 이것은 한국어 음성 테스트입니다.'
        }
        
        results = {}
        for lang_code, test_text in test_texts.items():
            lang_name = self.LANGUAGE_NAMES[lang_code]
            print(f"\n🔊 Testing {lang_name}...")
            
            success = self.text_to_speech(test_text, lang_code)
            results[lang_code] = success
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"   {lang_name}: {status}")
            
            time.sleep(1)
        
        successful = sum(results.values())
        print(f"\n📈 SUMMARY: {successful}/{len(results)} languages working")
        return results


# Create global instance for easy access
_global_tts_instance = None

def get_tts_engine():
    """Get or create global TTS engine instance"""
    global _global_tts_instance
    if _global_tts_instance is None:
        _global_tts_instance = MultilingualTextToSpeech()
    return _global_tts_instance

# Backward compatibility alias
TextToSpeech = MultilingualTextToSpeech

def test_voice_quality():
    """Test voice quality with simple sentences"""
    print("🔊 TESTING VOICE QUALITY")
    print("=" * 30)
    
    tts = MultilingualTextToSpeech()
    
    # Simple test sentences
    tests = [
        ('en', 'Hello. How are you?'),
        ('en', 'This is a clear voice test.'),
        ('hi', 'नमस्ते। आप कैसे हैं?'),
        ('es', 'Hola. ¿Cómo estás?'),
    ]
    
    for i, (lang, text) in enumerate(tests, 1):
        print(f"\n🎯 Test {i}: {tts.LANGUAGE_NAMES[lang]}")
        print(f"📝 '{text}'")
        print("Waiting...")
        time.sleep(2)
        
        success = tts.speak_text(text, lang)
        print(f"Result: {'✅ CLEAR' if success else '❌ BROKEN'}")
        
        if i < len(tests):
            time.sleep(3)
    
    print("\n🎯 VOICE QUALITY TEST COMPLETED")

if __name__ == "__main__":
    # Test voice quality first
    test_voice_quality()
    
    # Then comprehensive test
    tts = MultilingualTextToSpeech()
    if tts.is_available():
        print("\n" + "="*50)
        tts.test_all_languages()