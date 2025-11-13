#realtime_pipeline.py


import os
import time
import threading
import queue
import json
from datetime import datetime

class RealTimeTranslationPipeline:
    def __init__(self):
        print(" INITIALIZING REAL-TIME TRANSLATION PIPELINE")
        print("=" * 60)
        
        try:
            # Import existing modules with error handling
            from speech_recognizer import EnhancedSpeechRecognizer
            from translator import EnhancedTranslator
            
            self.speech_recognizer = EnhancedSpeechRecognizer()
            self.translator = EnhancedTranslator()
            
            # Get supported languages safely
            if hasattr(self.translator, 'supported_languages'):
                self.supported_languages = self.translator.supported_languages
            else:
                # Fallback language mapping
                self.supported_languages = {
                    'en': 'English', 'hi': 'Hindi', 'es': 'Spanish', 'fr': 'French',
                    'de': 'German', 'ja': 'Japanese', 'zh': 'Chinese', 'it': 'Italian',
                    'ru': 'Russian', 'ar': 'Arabic', 'ko': 'Korean', 'pt': 'Portuguese'
                }
                
        except Exception as e:
            print(f" Failed to initialize pipeline components: {e}")
            raise
        
        # Real-time processing queues
        self.speech_queue = queue.Queue()
        self.translation_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        # Pipeline control
        self.is_running = False
        self.processing_threads = []
        
        # OTT integration settings
        self.ott_enabled = False
        self.target_languages = ['hi', 'es', 'fr', 'de', 'ja', 'zh']
        
        # Performance metrics
        self.metrics = {
            'total_processed': 0,
            'avg_latency': 0,
            'success_rate': 0,
            'start_time': None
        }
        
        print(" Real-time pipeline initialized")
        print(f" Supported languages: {len(self.supported_languages)}")
    
    def start_realtime_session(self, source_type="microphone", target_languages=None):
        """Start real-time speech-to-speech translation session"""
        print(f"\n STARTING REAL-TIME SESSION: {source_type.upper()}")
        print("=" * 50)
        
        if target_languages:
            self.target_languages = target_languages
        
        self.metrics['start_time'] = time.time()
        self.is_running = True
        
        # Start processing threads
        self._start_processing_threads()
        
        print(" Real-time session started")
        print(f" Target languages: {[self.supported_languages.get(l, l) for l in self.target_languages]}")
        print(f" OTT Integration: {'ENABLED' if self.ott_enabled else 'DISABLED'}")
        
        return True
    
    def _start_processing_threads(self):
        """Start all processing threads for real-time pipeline"""
        print(" Starting real-time processing threads...")
        
        # Speech recognition thread
        speech_thread = threading.Thread(target=self._speech_processing_worker, daemon=True)
        speech_thread.start()
        self.processing_threads.append(speech_thread)
        
        # Translation thread
        translation_thread = threading.Thread(target=self._translation_processing_worker, daemon=True)
        translation_thread.start()
        self.processing_threads.append(translation_thread)
        
        # Output thread
        output_thread = threading.Thread(target=self._output_processing_worker, daemon=True)
        output_thread.start()
        self.processing_threads.append(output_thread)
        
        print(f" Started {len(self.processing_threads)} processing threads")
    
    def _speech_processing_worker(self):
        """Worker thread for speech recognition"""
        print(" Speech processing worker started")
        while self.is_running:
            try:
                # Check for speech input every 0.5 seconds
                time.sleep(0.5)
                
            except Exception as e:
                print(f" Speech processing error: {e}")
    
    def _translation_processing_worker(self):
        """Worker thread for translation"""
        print(" Translation processing worker started")
        while self.is_running:
            try:
                if not self.speech_queue.empty():
                    speech_data = self.speech_queue.get(timeout=1)
                    
                    # Translate to all target languages
                    source_text = speech_data['text']
                    source_lang = speech_data['language']
                    
                    start_time = time.time()
                    
                   
                    translations = {}
                    if hasattr(self.translator, 'batch_translate'):
                        translations = self.translator.batch_translate(
                            source_text, source_lang, self.target_languages
                        )
                    else:
                        
                        for target_lang in self.target_languages:
                            if target_lang != source_lang:
                                try:
                                    translated_text = self.translator.translate(source_text, source_lang, target_lang)
                                    translations[target_lang] = translated_text
                                except Exception as e:
                                    print(f"❌ Translation error for {target_lang}: {e}")
                                    translations[target_lang] = f"[Error: {e}]"
                    
                    latency = time.time() - start_time
                    
                    # Update metrics
                    self.metrics['total_processed'] += 1
                    if self.metrics['total_processed'] == 1:
                        self.metrics['avg_latency'] = latency
                    else:
                        self.metrics['avg_latency'] = (
                            self.metrics['avg_latency'] * (self.metrics['total_processed'] - 1) + latency
                        ) / self.metrics['total_processed']
                    
                    # Put in output queue
                    self.translation_queue.put({
                        'original': speech_data,
                        'translations': translations,
                        'latency': latency,
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f" Translation processing error: {e}")
    
    def _output_processing_worker(self):
        """Worker thread for output and OTT integration"""
        print(" Output processing worker started")
        while self.is_running:
            try:
                if not self.translation_queue.empty():
                    translation_data = self.translation_queue.get(timeout=1)
                    
                    # Save to database
                    self._save_translation_output(translation_data)
                    
                    # Display real-time results
                    self._display_realtime_output(translation_data)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f" Output processing error: {e}")
    
    def _save_translation_output(self, translation_data):
        """Save translation results to database"""
        try:
            original = translation_data['original']
            translations = translation_data['translations']
            
            # Save transcript
            try:
                from db_json import save_line_transcript
                save_line_transcript(
                    original['text'],
                    original['language'],
                    self.metrics['total_processed'],
                    "realtime_pipeline"
                )
            except Exception as e:
                print(f" Could not save transcript: {e}")
            
            # Save translations
            try:
                from db_json import save_line_translation
                save_line_translation(
                    original['text'],
                    original['language'],
                    translations,
                    self.metrics['total_processed'],
                    "realtime_pipeline"
                )
            except Exception as e:
                print(f" Could not save translations: {e}")
            
        except Exception as e:
            print(f" Error saving translation output: {e}")
    
    def _display_realtime_output(self, translation_data):
        """Display real-time translation results"""
        original = translation_data['original']
        translations = translation_data['translations']
        
        print(f"\n REAL-TIME TRANSLATION #{self.metrics['total_processed']}")
        print(f" Original ({original['language']}): {original['text']}")
        print(f" Latency: {translation_data['latency']:.3f}s")
        
        successful_translations = 0
        for lang, text in translations.items():
            if text and not text.startswith('[Error'):  # Only show successful translations
                lang_name = self.supported_languages.get(lang, lang)
                print(f"    {lang_name}: {text}")
                successful_translations += 1
        
        if successful_translations == 0:
            print("    No successful translations")
        
        print(f" Metrics: {self.metrics['total_processed']} processed, "
              f"Avg latency: {self.metrics['avg_latency']:.3f}s")
    
    def enable_ott_integration(self, enabled=True):
        """Enable/disable OTT feed integration"""
        self.ott_enabled = enabled
        status = "ENABLED" if enabled else "DISABLED"
        print(f" OTT Integration: {status}")
    
    def set_target_languages(self, languages):
        """Set target languages for translation"""
        valid_languages = []
        
        for lang in languages:
            if lang in self.supported_languages:
                valid_languages.append(lang)
            else:
                print(f" Unsupported language skipped: {lang}")
        
        self.target_languages = valid_languages
        print(f" Target languages set: {[self.supported_languages.get(l, l) for l in valid_languages]}")
    
    def get_performance_metrics(self):
        """Get current performance metrics"""
        if self.metrics['start_time']:
            session_duration = time.time() - self.metrics['start_time']
            throughput = self.metrics['total_processed'] / session_duration if session_duration > 0 else 0
            self.metrics['throughput'] = throughput
        
        return self.metrics.copy()
    
    def stop_realtime_session(self):
        """Stop the real-time session"""
        print("\n STOPPING REAL-TIME SESSION")
        self.is_running = False
        
        # Wait for threads to finish
        for thread in self.processing_threads:
            thread.join(timeout=2)
        
        # Final metrics
        final_metrics = self.get_performance_metrics()
        print(f" FINAL SESSION METRICS:")
        print(f"   Total processed: {final_metrics['total_processed']}")
        print(f"   Average latency: {final_metrics['avg_latency']:.3f}s")
        if 'throughput' in final_metrics:
            print(f"   Throughput: {final_metrics['throughput']:.1f} translations/sec")
        
        return final_metrics

    def process_audio_stream(self, audio_file_path=None):
        """Process audio stream in real-time simulation"""
        print(f"\n PROCESSING AUDIO STREAM: {audio_file_path or 'Microphone'}")
        
        if audio_file_path and os.path.exists(audio_file_path):
            # Process audio file with real-time simulation
            return self._process_audio_file_realtime(audio_file_path)
        else:
            # Use microphone input
            return self._process_microphone_realtime()
    
    def _process_audio_file_realtime(self, file_path):
        """Process audio file with real-time simulation"""
        try:
            from speech_recognizer import EnhancedSpeechRecognizer
            
            recognizer = EnhancedSpeechRecognizer()
            print(" Processing audio file with real-time simulation...")
            
            # Get file info
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f" File size: {file_size:.1f}MB")
            
            # Process in chunks to simulate real-time
            text, language = recognizer.recognize_from_audio_file(file_path)
            
            if text and language:
                # Simulate real-time processing by breaking into sentences
                sentences = [s.strip() for s in text.split('.') if s.strip()]
                print(f" Detected {len(sentences)} sentences in {language}")
                
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        # Simulate real-time processing delay
                        time.sleep(1.0)
                        
                        # Process this sentence through the pipeline
                        speech_data = {
                            'text': sentence.strip(),
                            'language': language,
                            'timestamp': datetime.now().isoformat(),
                            'sentence_id': i + 1
                        }
                        
                        # Add to queue for processing
                        self.speech_queue.put(speech_data)
                        
                        print(f" Processed sentence {i+1}/{len(sentences)}")
                
                return True
            else:
                print(" No speech detected in audio file")
                return False
                
        except Exception as e:
            print(f" Audio file processing error: {e}")
            return False
    
    def _process_microphone_realtime(self):
        """Process microphone input in real-time"""
        print(" Starting real-time microphone processing...")
        
        try:
            from speech_recognizer import EnhancedSpeechRecognizer
            recognizer = EnhancedSpeechRecognizer()
            
            # Start real-time session
            self.start_realtime_session("microphone")
            
            print(" Speak into the microphone... (Press Ctrl+C to stop)")
            
            while self.is_running:
                try:
                    # Get microphone input
                    text, language = recognizer.recognize_from_microphone()
                    
                    if text and language:
                        # Add to processing queue
                        speech_data = {
                            'text': text,
                            'language': language,
                            'timestamp': datetime.now().isoformat()
                        }
                        self.speech_queue.put(speech_data)
                        print(f" Captured: {text[:50]}...")
                        
                    time.sleep(2)  # Check every 2 seconds to avoid spamming
                    
                except KeyboardInterrupt:
                    print("\n Microphone processing interrupted by user")
                    break
                except Exception as e:
                    print(f" Microphone processing error: {e}")
            
            # Stop the session
            self.stop_realtime_session()
            return True
            
        except Exception as e:
            print(f" Microphone real-time error: {e}")
            return False

    def add_speech_input(self, text, language='en'):
        """Manually add speech input for testing"""
        speech_data = {
            'text': text,
            'language': language,
            'timestamp': datetime.now().isoformat()
        }
        self.speech_queue.put(speech_data)
        print(f" Added speech input: '{text}' ({language})")
    
    def real_time_translate(self, text, source_lang, target_langs=None):
        """Real-time translation method - FIXED"""
        if target_langs is None:
            target_langs = ['hi', 'es', 'fr']  # Default target languages
        
        try:
            results = {}
            for lang in target_langs:
                if lang != source_lang:
                    try:
                        translated = self.translator.translate(text, source_lang, lang)
                        results[lang] = translated
                    except Exception as e:
                        print(f" Translation error for {lang}: {e}")
                        results[lang] = f"[Translation error: {e}]"
            return results
        except Exception as e:
            print(f" Real-time translation error: {e}")
            return {f"error_{lang}": f"[System error: {e}]" for lang in target_langs}

# Backward compatibility
RealTimePipeline = RealTimeTranslationPipeline

# Test function
def test_realtime_pipeline():
    """Test the real-time pipeline"""
    print("\n TESTING REAL-TIME PIPELINE")
    print("=" * 40)
    
    try:
        pipeline = RealTimeTranslationPipeline()
        
        # Test manual input
        print("\n Testing manual input...")
        pipeline.start_realtime_session("test")
        
        # Add test sentences
        test_sentences = [
            "Hello, how are you today?",
            "This is a test of the real-time translation system.",
            "The weather is nice today."
        ]
        
        for i, sentence in enumerate(test_sentences):
            pipeline.add_speech_input(sentence, 'en')
            time.sleep(2)  # Wait for processing
        
        # Let it run for a bit
        time.sleep(5)
        
        # Stop and show metrics
        metrics = pipeline.stop_realtime_session()
        print(f"\n Test completed: {metrics['total_processed']} translations processed")
        
        return True
        
    except Exception as e:
        print(f" Pipeline test failed: {e}")
        return False

if __name__ == "__main__":
    test_realtime_pipeline()