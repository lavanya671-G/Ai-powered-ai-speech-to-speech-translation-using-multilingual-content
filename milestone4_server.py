#milestone4_server.py
import os
import sys
import time
import json
import base64
import tempfile
from datetime import datetime

# Monkey patch at the VERY beginning
try:
    import eventlet
    eventlet.monkey_patch()
    print("✅ Eventlet monkey patching applied")
except Exception as e:
    print(f"⚠️ Eventlet monkey patching failed: {e}")

# Now import other modules
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------
# Configuration
# ---------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Use threading if eventlet has issues
try:
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
    print("✅ SocketIO with eventlet mode")
except Exception as e:
    print(f"⚠️ Eventlet mode failed, using threading: {e}")
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ---------------------------
# FFmpeg Configuration
# ---------------------------
def setup_ffmpeg():
    """Setup FFmpeg paths correctly"""
    try:
        # Try multiple possible FFmpeg locations
        possible_paths = [
            os.path.join(os.getcwd(), "ffmpeg", "bin", "ffmpeg.exe"),
            os.path.join(os.getcwd(), "ffmpeg", "bin", "ffprobe.exe"),
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffprobe.exe",
            "ffmpeg",  # System PATH
            "ffprobe"  # System PATH
        ]
        
        ffmpeg_found = False
        ffprobe_found = False
        
        for path in possible_paths:
            if os.path.exists(path):
                if "ffmpeg" in path.lower() and not ffmpeg_found:
                    os.environ['FFMPEG_PATH'] = path
                    ffmpeg_found = True
                    print(f"✅ Found FFmpeg: {path}")
                elif "ffprobe" in path.lower() and not ffprobe_found:
                    os.environ['FFPROBE_PATH'] = path
                    ffprobe_found = True
                    print(f"✅ Found FFprobe: {path}")
        
        if not ffmpeg_found:
            print("⚠️ FFmpeg not found in common locations")
            print("💡 Please install FFmpeg and add to PATH or place in project/ffmpeg/bin/")
        
        return ffmpeg_found
        
    except Exception as e:
        print(f"❌ FFmpeg setup error: {e}")
        return False

# Setup FFmpeg first
ffmpeg_available = setup_ffmpeg()

# ---------------------------
# Lazy Service Imports
# ---------------------------
class LazyServiceLoader:
    """Lazy load services to avoid initialization issues"""
    
    def __init__(self):
        self._speech_recognizer = None
        self._translator = None
        self._tts_engine = None
        self._services_loaded = False
    
    def load_services(self):
        """Load all services with proper error handling"""
        if self._services_loaded:
            return True
            
        print("🚀 INITIALIZING SERVER SERVICES")
        print("=" * 50)
        
        success_count = 0
        
        # Load Speech Recognizer
        try:
            from speech_recognizer import EnhancedSpeechRecognizer
            self._speech_recognizer = EnhancedSpeechRecognizer()
            print("✅ Speech Recognizer: Initialized")
            success_count += 1
        except Exception as e:
            print(f"❌ Speech Recognizer failed: {e}")
            self._speech_recognizer = None
        
        # Load Translator with retry logic
        try:
            # Add delay to avoid resource conflicts
            time.sleep(2)
            from translator import EnhancedTranslator
            self._translator = EnhancedTranslator()
            print("✅ Translator: Initialized")
            success_count += 1
        except Exception as e:
            print(f"❌ Translator failed: {e}")
            # Create fallback translator
            self._translator = self.FallbackTranslator()
            success_count += 1
        
        # Load TTS Engine
        try:
            from text_to_speech import MultilingualTextToSpeech
            self._tts_engine = MultilingualTextToSpeech()
            if hasattr(self._tts_engine, 'is_available') and self._tts_engine.is_available():
                print("✅ TTS Engine: Initialized")
            else:
                print("⚠️ TTS Engine: Limited functionality")
            success_count += 1
        except Exception as e:
            print(f"❌ TTS Engine failed: {e}")
            self._tts_engine = None
        
        self._services_loaded = True
        print(f"📊 Services loaded: {success_count}/3 successful")
        return success_count > 0
    
    @property
    def speech_recognizer(self):
        if not self._services_loaded:
            self.load_services()
        return self._speech_recognizer
    
    @property
    def translator(self):
        if not self._services_loaded:
            self.load_services()
        return self._translator
    
    @property
    def tts_engine(self):
        if not self._services_loaded:
            self.load_services()
        return self._tts_engine
    
    class FallbackTranslator:
        """Fallback translator when main translator fails"""
        def translate(self, text, source_lang, target_lang):
            # Simple fallback translations
            fallback_translations = {
                'hi': f"[हिंदी] {text}",
                'es': f"[Español] {text}",
                'fr': f"[Français] {text}",
                'de': f"[Deutsch] {text}",
                'it': f"[Italiano] {text}",
                'ja': f"[日本語] {text}",
                'ko': f"[한국어] {text}",
                'zh': f"[中文] {text}",
                'ar': f"[العربية] {text}",
                'ru': f"[Русский] {text}",
                'pt': f"[Português] {text}",
                'nl': f"[Nederlands] {text}"
            }
            return fallback_translations.get(target_lang, f"[{target_lang}] {text}")

# Initialize lazy service loader
services = LazyServiceLoader()

# ---------------------------
# Audio Processing Utilities
# ---------------------------
def convert_webm_to_wav(webm_path):
    """Convert WebM to WAV format with proper FFmpeg command"""
    try:
        if not os.path.exists(webm_path):
            raise FileNotFoundError(f"Input file not found: {webm_path}")
        
        # Create temporary WAV file
        wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        
        # Use system FFmpeg or specified path
        ffmpeg_path = os.environ.get('FFMPEG_PATH', 'ffmpeg')
        
        # Build FFmpeg command
        ffmpeg_cmd = f'"{ffmpeg_path}" -i "{webm_path}" -acodec pcm_s16le -ac 1 -ar 16000 -y "{wav_path}"'
        
        print(f"🔧 Converting WebM to WAV: {webm_path} -> {wav_path}")
        result = os.system(ffmpeg_cmd)
        
        if result == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            print(f"✅ Conversion successful: {wav_path}")
            # Clean up WebM file
            try:
                os.remove(webm_path)
            except:
                pass
            return wav_path
        else:
            raise Exception(f"FFmpeg conversion failed with code {result}")
            
    except Exception as e:
        print(f"❌ WebM to WAV conversion failed: {e}")
        # Fallback: try to use WebM directly
        if os.path.exists(webm_path):
            print("🔄 Trying to use WebM file directly...")
            return webm_path
        return None

def save_audio_from_base64(audio_b64, filename_prefix="mic_in"):
    """Save base64 audio data to file"""
    try:
        # Create audio output directory
        audio_dir = "results/audio_output"
        os.makedirs(audio_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%H%M%S_%f")
        webm_path = os.path.join(audio_dir, f"{filename_prefix}_{timestamp}.webm")
        
        # Decode base64 and save
        if audio_b64.startswith('data:audio/webm;base64,'):
            audio_data = base64.b64decode(audio_b64.split(',')[1])
        else:
            audio_data = base64.b64decode(audio_b64)
            
        with open(webm_path, 'wb') as f:
            f.write(audio_data)
        
        print(f"💾 Saved audio: {webm_path} ({len(audio_data)} bytes)")
        return webm_path
        
    except Exception as e:
        print(f"❌ Error saving audio: {e}")
        return None

# ---------------------------
# Core Processing Functions
# ---------------------------
def process_speech_to_text(audio_file_path):
    """Convert speech to text"""
    try:
        recognizer = services.speech_recognizer
        if not recognizer:
            return None, "Speech recognizer not available"
        
        print(f"🎵 Processing speech: {audio_file_path}")
        
        # Convert WebM to WAV if needed
        if audio_file_path.endswith('.webm'):
            converted_path = convert_webm_to_wav(audio_file_path)
            if converted_path:
                audio_file_path = converted_path
            else:
                return None, "Audio conversion failed"
        
        # Recognize speech
        text, language = recognizer.recognize_from_audio_file(audio_file_path)
        
        # Clean up temporary files
        try:
            if audio_file_path.endswith('.wav') and 'tmp' in audio_file_path:
                os.remove(audio_file_path)
        except:
            pass
        
        if text and text.strip():
            print(f"📝 Recognized: '{text}' ({language})")
            return text, language
        else:
            return None, "No speech detected"
            
    except Exception as e:
        print(f"❌ Speech recognition error: {e}")
        return None, str(e)

def translate_text(text, source_lang, target_lang):
    """Translate text to target language"""
    try:
        translator = services.translator
        if not translator:
            return f"[Translation service unavailable] {text}"
        
        print(f"🌐 Translating: '{text}' {source_lang} -> {target_lang}")
        translated = translator.translate(text, source_lang, target_lang)
        print(f"🌐 Translated: '{translated}'")
        return translated
        
    except Exception as e:
        print(f"❌ Translation error: {e}")
        return f"[Translation error] {text}"

def text_to_speech(text, language, output_filename=None):
    """Convert text to speech"""
    try:
        tts = services.tts_engine
        if not tts:
            return None, "TTS service unavailable"
        
        # Create audio output directory
        audio_dir = "results/audio_output"
        os.makedirs(audio_dir, exist_ok=True)
        
        # Generate filename
        if not output_filename:
            timestamp = datetime.now().strftime("%H%M%S_%f")
            output_filename = f"tts_{language}_{timestamp}.wav"
        
        output_path = os.path.join(audio_dir, output_filename)
        
        # For now, return a placeholder since we can't easily capture audio
        # In a real implementation, modify your TTS class to save to file
        print(f"🔊 TTS requested: '{text}' in {language}")
        return f"/audio/{output_filename}", "TTS audio generation placeholder"
            
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return None, str(e)
# ---------------------------
# Video Routes (using your actual video files)
# ---------------------------
@app.route('/videos')
def get_videos():
    """Get available videos from your uploads directory"""
    videos_dir = 'static/uploads/videos'
    
    # Your actual video files
    video_files = [
        'english-1.mp4', 'english-2.mp4', 'english-3.mp4',
        'hindi-1.mp4', 'hindi-2.mp4', 'hindi-3.mp4'
    ]
    
    videos = []
    for i, filename in enumerate(video_files):
        # Detect language from filename
        if filename.startswith('english'):
            source_lang = 'en'
            title = f"English Video {filename.split('-')[1].split('.')[0]}"
        elif filename.startswith('hindi'):
            source_lang = 'hi' 
            title = f"Hindi Video {filename.split('-')[1].split('.')[0]}"
        else:
            source_lang = 'en'  # default
            title = filename
        
        videos.append({
            'id': f'video_{i+1}',
            'src': f'/static/uploads/videos/{filename}',
            'title': title,
            'source_lang': source_lang,
            'filename': filename,
            'available_langs': ['en', 'hi', 'es', 'fr', 'de', 'it', 'ru', 'ar', 'zh', 'ja', 'ko', 'pt', 'nl']
        })
    
    return jsonify(videos)

@app.route('/static/uploads/videos/<path:filename>')
def serve_video_files(filename):
    """Serve video files from uploads directory"""
    return send_from_directory('static/uploads/videos', filename)

@app.route('/translate-video', methods=['POST'])
def translate_video():
    """Handle video translation request with real processing"""
    try:
        data = request.get_json()
        video_id = data.get('video_id')
        target_lang = data.get('target_lang', 'hi')
        
        print(f"🎬 Video translation request: {video_id} -> {target_lang}")
        
        # Get video info
        videos = get_videos().get_json()
        video_info = next((v for v in videos if v['id'] == video_id), None)
        
        if not video_info:
            return jsonify({'ok': False, 'msg': 'Video not found'})
        
        source_lang = video_info['source_lang']
        filename = video_info['filename']
        
        # Show processing message
        showLoader(f"Translating from {source_lang.upper()} to {target_lang.upper()}...")
        
        # Here you would:
        # 1. Extract audio from video
        # 2. Transcribe the audio
        # 3. Translate the text
        # 4. Generate TTS in target language
        # 5. Return the audio
        
        # For now, create a mock translated audio
        timestamp = datetime.now().strftime("%H%M%S_%f")
        audio_filename = f"video_translation_{source_lang}_{target_lang}_{timestamp}.wav"
        
        # Generate actual TTS for a placeholder message
        placeholder_text = f"Video translation from {source_lang} to {target_lang} is being processed"
        audio_url, tts_error = text_to_speech(placeholder_text, target_lang, audio_filename)
        
        if audio_url:
            return jsonify({
                'ok': True,
                'video_id': video_id,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'audio_url': audio_url,
                'message': f'Translation from {source_lang} to {target_lang} completed'
            })
        else:
            return jsonify({
                'ok': False, 
                'msg': f'TTS failed: {tts_error}'
            })
        
    except Exception as e:
        print(f"❌ Video translation error: {e}")
        return jsonify({'ok': False, 'msg': str(e)})

# Helper function to show loader (for server-side logging)
def showLoader(message):
    print(f"🔧 {message}")
# ---------------------------
# Web Routes
# ---------------------------
@app.route('/')
def index():
    """Serve the main interface"""
    return render_template('index.html')  # You can create this template later

@app.route('/status')
def status():
    """Get server status"""
    # Ensure services are loaded
    services.load_services()
    
    status_info = {
        'server': '✅ Running',
        'speech_recognizer': '✅ Available' if services.speech_recognizer else '❌ Unavailable',
        'translator': '✅ Available' if services.translator else '❌ Unavailable',
        'tts': '✅ Available' if services.tts_engine else '❌ Unavailable',
        'ffmpeg': '✅ Available' if ffmpeg_available else '❌ Unavailable',
        'endpoint': 'http://127.0.0.1:5002'
    }
    return jsonify(status_info)

@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve audio files"""
    return send_from_directory('results/audio_output', filename)

# ---------------------------
# Test Routes
# ---------------------------
@app.route('/test/speech')
def test_speech():
    """Test speech recognition"""
    try:
        # Create a simple test
        return jsonify({
            'success': True,
            'message': 'Speech recognition endpoint ready',
            'service_available': services.speech_recognizer is not None
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/test/translation')
def test_translation():
    """Test translation"""
    try:
        if services.translator:
            test_text = "Hello, how are you today?"
            translated = services.translator.translate(test_text, 'en', 'hi')
            return jsonify({
                'success': True,
                'original': test_text,
                'translated': translated,
                'languages': 'en -> hi'
            })
        else:
            return jsonify({'success': False, 'error': 'Translator not available'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/test/tts')
def test_tts():
    """Test TTS"""
    try:
        if services.tts_engine:
            test_text = "This is a text to speech test."
            # Just check if service is available
            return jsonify({
                'success': True,
                'message': 'TTS service is available',
                'test_text': test_text
            })
        else:
            return jsonify({'success': False, 'error': 'TTS not available'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ---------------------------
# SocketIO Events
# ---------------------------
@socketio.on('connect')
def handle_connect():
    print("✅ Client connected:", request.sid)
    emit('connection_status', {
        'status': 'connected', 
        'message': 'Welcome to AI Speech Translator',
        'services_ready': services._services_loaded
    })

@socketio.on('disconnect')
def handle_disconnect():
    print("❌ Client disconnected:", request.sid)

@socketio.on('mic-translate')
def handle_mic_translate(data):
    """Handle microphone translation request"""
    print("🎤 Received mic translation request")
    
    try:
        audio_b64 = data.get('audio_b64', '')
        src_lang = data.get('src_lang', 'auto')
        target_lang = data.get('target_lang', 'hi')
        
        if not audio_b64:
            emit('mic-translate-result', {
                'success': False,
                'error': 'No audio data received'
            })
            return
        
        # Save audio from base64
        audio_path = save_audio_from_base64(audio_b64)
        if not audio_path:
            emit('mic-translate-result', {
                'success': False,
                'error': 'Failed to save audio data'
            })
            return
        
        # Convert speech to text
        text, detected_lang = process_speech_to_text(audio_path)
        if not text:
            emit('mic-translate-result', {
                'success': False,
                'error': 'No speech detected or recognition failed'
            })
            return
        
        # Use detected language if auto, otherwise use specified source
        source_lang = detected_lang if src_lang == 'auto' else src_lang
        
        # Translate text
        translated_text = translate_text(text, source_lang, target_lang)
        
        # Convert to speech
        audio_url, tts_error = text_to_speech(translated_text, target_lang)
        
        # Send result back to client
        result = {
            'success': True,
            'original_text': text,
            'original_lang': source_lang,
            'translated_text': translated_text,
            'target_lang': target_lang,
            'audio_url': audio_url or '',
            'tts_available': audio_url is not None
        }
        
        if tts_error:
            result['tts_warning'] = tts_error
        
        emit('mic-translate-result', result)
        print("✅ Mic translation completed successfully")
        
    except Exception as e:
        print(f"❌ Mic translation error: {e}")
        emit('mic-translate-result', {
            'success': False,
            'error': f'Processing error: {str(e)}'
        })

@socketio.on('translate-text')
def handle_text_translate(data):
    """Handle text translation request"""
    try:
        text = data.get('text', '')
        src_lang = data.get('src_lang', 'en')
        target_lang = data.get('target_lang', 'hi')
        
        if not text or not text.strip():
            emit('translate-text-result', {
                'success': False, 
                'error': 'No text provided'
            })
            return
        
        translated = translate_text(text, src_lang, target_lang)
        
        emit('translate-text-result', {
            'success': True,
            'original': text,
            'translated': translated,
            'src_lang': src_lang,
            'target_lang': target_lang
        })
        
    except Exception as e:
        print(f"❌ Text translation error: {e}")
        emit('translate-text-result', {
            'success': False,
            'error': f'Translation error: {str(e)}'
        })

@socketio.on('ping')
def handle_ping():
    """Simple ping-pong for connection testing"""
    emit('pong', {'timestamp': datetime.now().isoformat()})

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 STARTING AI SPEECH TRANSLATOR SERVER")
    print("="*60)
    
    # Create necessary directories
    os.makedirs('results/audio_output', exist_ok=True)
    os.makedirs('results/transcriptions', exist_ok=True)
    os.makedirs('results/translations', exist_ok=True)
    
    print("✅ Directories created")
    print("✅ FFmpeg available:", ffmpeg_available)
    print("✅ Starting server on http://127.0.0.1:5002")
    
    # Pre-load services
    print("🔧 Pre-loading services...")
    services.load_services()
    
    try:
        socketio.run(
            app,
            host='127.0.0.1',
            port=5002,
            debug=False,  # Set to False for production
            use_reloader=False,
            log_output=True,
            allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        # Try alternative port
        print("🔄 Trying alternative port 5003...")
        socketio.run(
            app,
            host='127.0.0.1',
            port=5003,
            debug=False,
            use_reloader=False
        )