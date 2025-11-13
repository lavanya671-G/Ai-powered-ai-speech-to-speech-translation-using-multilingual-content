# ================================================================
# speech_recognizer.py - FIXED VERSION WITH PROPER FFMPEG HANDLING
# ================================================================
import os
import math
import tempfile
import time
import subprocess
from pydub import AudioSegment, silence
from dotenv import load_dotenv
import ffmpeg
import uuid

# --- Force local FFmpeg path ---
FFMPEG_PATH = os.path.join(os.getcwd(), "ffmpeg", "bin", "ffmpeg.exe")
FFPROBE_PATH = os.path.join(os.getcwd(), "ffmpeg", "bin", "ffprobe.exe")

if os.path.exists(FFMPEG_PATH):
    AudioSegment.converter = FFMPEG_PATH
    if os.path.exists(FFPROBE_PATH):
        AudioSegment.ffprobe = FFPROBE_PATH
    print(f" Using local FFmpeg: {FFMPEG_PATH}")
else:
    print(" Local FFmpeg not found, using system FFmpeg")

# Try Azure SDK
try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print(" Azure Speech SDK not installed — STT disabled.")

# Import JSON save helpers (safe import)
try:
    from db_json import save_line_transcript, save_final_transcript
except ImportError:
    def save_line_transcript(*args, **kwargs): 
        print(f" Transcript would save: {args}")
    def save_final_transcript(*args, **kwargs): 
        print(f"Final transcript would save: {args}")

# Load .env and get Azure keys
load_dotenv()
speech_key = os.getenv("SPEECH_KEY")
speech_region = os.getenv("SPEECH_REGION")

# Validate Azure credentials
if AZURE_AVAILABLE:
    if not speech_key or not speech_region:
        print(" MISSING Azure Speech Key or Region in .env")
        print(" Please check your .env file has SPEECH_KEY and SPEECH_REGION")
        AZURE_AVAILABLE = False
    else:
        print(f" Azure credentials loaded - Region: {speech_region}")

# --- Constants ---
CHUNK_LENGTH_MS = 30_000  # Reduced to 30 seconds for better reliability

def run_ffmpeg_command(input_path, output_path):
    """Run FFmpeg command using subprocess (Windows compatible)"""
    try:
        # Build FFmpeg command as list
        ffmpeg_cmd = [
            FFMPEG_PATH,
            "-i", input_path,
            "-ar", "16000",    # Sample rate
            "-ac", "1",        # Mono audio
            "-acodec", "pcm_s16le",  # Audio codec
            "-y",              # Overwrite output
            output_path
        ]
        
        print(f" Running FFmpeg: {' '.join(ffmpeg_cmd)}")
        
        # Use subprocess with timeout
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0  # No console window on Windows
        )
        
        if result.returncode == 0:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f" FFmpeg conversion successful: {output_path}")
                return True
            else:
                print(" FFmpeg: Output file missing or empty")
                return False
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown FFmpeg error"
            print(f" FFmpeg failed: {error_msg}")
            return False
            
    except subprocess.TimeoutExpired:
        print(" FFmpeg conversion timeout (30 seconds)")
        return False
    except Exception as e:
        print(f" FFmpeg subprocess error: {e}")
        return False


def convert_webm_to_wav(input_path, output_folder="results/temp_audio"):
    out_name = f"{uuid.uuid4().hex}.wav"
    out_path = os.path.join(output_folder, out_name)

    (
        ffmpeg
        .input(input_path)
        .output(out_path, ac=1, ar=16000, format='wav')
        .overwrite_output()
        .run(quiet=True)
    )

    return out_path


def split_audio_to_chunks(file_path, min_silence_len=1500, silence_thresh=-40, chunk_length_ms=CHUNK_LENGTH_MS):
    """Split audio intelligently for better STT performance - FIXED FFMPEG"""
    try:
        print(f" Loading audio file: {file_path}")
        
        # Validate file exists and has content
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError("Audio file is empty")
        
        print(f" File size: {file_size} bytes")
        
        # Check file extension for special handling
        file_ext = os.path.splitext(file_path)[1].lower()
        print(f" File extension: {file_ext}")
        
        # Handle MP4 files specially
        if file_ext == '.mp4':
            print(" MP4 detected - using direct FFmpeg conversion")
            tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            if run_ffmpeg_command(file_path, tmp_wav):
                return [tmp_wav]
            else:
                raise Exception("MP4 to WAV conversion failed")
        
        # Try to load audio with multiple format support
        try:
            print(" Attempting direct audio load...")
            audio = AudioSegment.from_file(file_path)
        except Exception as e:
            print(f" Primary audio load failed: {e}, trying FFmpeg fallback...")
            # FFmpeg fallback conversion - FIXED
            tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            if run_ffmpeg_command(file_path, tmp_wav):
                audio = AudioSegment.from_file(tmp_wav)
            else:
                raise Exception(f"FFmpeg conversion failed for {file_path}")
        
        duration_ms = len(audio)
        print(f" Audio duration: {duration_ms/1000:.2f}s")

        # For short files, no splitting needed
        if duration_ms <= 45000:  # 45 seconds
            print(" Short file — no splitting needed")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                audio.export(tmp.name, format="wav", parameters=["-ac", "1", "-ar", "16000"])
                return [tmp.name]

        # Try silence-based splitting first
        print(" Analyzing silence patterns...")
        chunks = silence.split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=800  # Increased keep_silence for better segmentation
        )

        chunk_paths = []
        if not chunks or len(chunks) == 0:
            print(" No silence detected, using fixed-length chunks")
            num_chunks = math.ceil(duration_ms / chunk_length_ms)
            for i in range(num_chunks):
                start = i * chunk_length_ms
                end = min((i + 1) * chunk_length_ms, duration_ms)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    chunk = audio[start:end]
                    chunk.export(tmp.name, format="wav", parameters=["-ac", "1", "-ar", "16000"])
                    chunk_paths.append(tmp.name)
                    print(f" Created fixed chunk {i+1}: {len(chunk)/1000:.2f}s")
        else:
            print(f" Found {len(chunks)} chunks based on silence")
            for i, chunk in enumerate(chunks):
                if len(chunk) > 2000:  # At least 2 seconds
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        chunk.export(tmp.name, format="wav", parameters=["-ac", "1", "-ar", "16000"])
                        chunk_paths.append(tmp.name)
                        print(f" Created silence-based chunk {i+1}: {len(chunk)/1000:.2f}s")

        print(f" Total chunks created: {len(chunk_paths)}")
        return chunk_paths

    except Exception as e:
        print(f" Audio splitting error: {e}")
        print(" Trying direct FFmpeg conversion as last resort...")
        try:
            tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            if run_ffmpeg_command(file_path, tmp_wav):
                return [tmp_wav]
            else:
                raise Exception("Final FFmpeg conversion failed")
        except Exception as err:
            print(f" All conversion methods failed: {err}")
        return []

def normalize_azure_lang(code):
    """Convert Azure 'en-US' -> 'en'"""
    if not code:
        return "en"
    return code.split("-")[0].lower()

class EnhancedSpeechRecognizer:
    def __init__(self):
        if not AZURE_AVAILABLE:
            raise ImportError("Azure Speech SDK not available")
            
        if not speech_key or not speech_region:
            raise ValueError(" Missing Azure Speech Key or Region in .env")

        try:
            self.speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
            self.speech_config.speech_recognition_language = "en-US"
            
            # Configure for better performance
            self.speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "5000")
            self.speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "2000")
            
            self.auto_detect_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                languages=["en-US", "hi-IN", "es-ES", "fr-FR"]
            )
            print(" Azure Speech Recognizer configured successfully")
            
        except Exception as e:
            print(f" Failed to configure Azure Speech: {e}")
            raise

    def recognize_from_audio_file(self, file_path):
        """Recognize speech from audio file with IMPROVED MP4 support"""
        print(f" Recognizing speech from: {file_path}")

        if not os.path.exists(file_path):
            print(f" File not found: {file_path}")
            return "", "en-US"

        # Check if it's MP4 and handle specially
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.mp4':
            print(" MP4 file detected - using enhanced processing...")
            return self._process_mp4_file(file_path)

        try:
            chunk_files = split_audio_to_chunks(file_path)
            if not chunk_files:
                print(" No audio chunks could be created")
                return "", "en-US"

            return self._process_audio_chunks(chunk_files, file_path)

        except Exception as e:
            print(f" Recognition failed: {e}")
            return "", "en-US"

    def _process_mp4_file(self, mp4_path):
        """Special processing for MP4 files"""
        try:
            print(" Extracting audio from MP4...")
            
            # Create temporary WAV file
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            
            # Use enhanced FFmpeg for MP4
            ffmpeg_cmd = [
                FFMPEG_PATH,
                "-i", mp4_path,
                "-vn",              # No video
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                "-y",               # Overwrite
                temp_wav
            ]
            
            print(f" Running MP4 extraction: {' '.join(ffmpeg_cmd)}")
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=60,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            if result.returncode != 0 or not os.path.exists(temp_wav) or os.path.getsize(temp_wav) == 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                print(f" MP4 audio extraction failed: {error_msg}")
                return "", "en-US"
            
            print(f" MP4 audio extracted to: {temp_wav}")
            
            # Process the extracted WAV file
            chunk_files = split_audio_to_chunks(temp_wav)
            
            if not chunk_files:
                print(" No audio chunks could be created from extracted audio")
                # Clean up
                try:
                    os.unlink(temp_wav)
                except:
                    pass
                return "", "en-US"

            result = self._process_audio_chunks(chunk_files, mp4_path)
            
           
            try:
                os.unlink(temp_wav)
            except:
                pass
                
            return result

        except Exception as e:
            print(f" MP4 processing failed: {e}")
            return "", "en-US"

    def _process_audio_chunks(self, chunk_files, original_file_path):
        """Process audio chunks and return results"""
        all_texts = []
        detected_lang = None

        for i, chunk_path in enumerate(chunk_files):
            try:
                print(f" Processing chunk {i+1}/{len(chunk_files)}: {os.path.basename(chunk_path)}")
                
                if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) == 0:
                    print(f" Chunk {i+1} is empty or missing, skipping")
                    continue

                audio_config = speechsdk.audio.AudioConfig(filename=chunk_path)
                recognizer = speechsdk.SpeechRecognizer(
                    speech_config=self.speech_config,
                    auto_detect_source_language_config=self.auto_detect_config,
                    audio_config=audio_config
                )

                # Use recognize_once instead of async for simplicity
                result = recognizer.recognize_once()
                
                if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    # Get detected language
                    lang_result = result.properties.get(
                        speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult, "en-US"
                    )
                    detected_lang = detected_lang or lang_result
                    text = result.text.strip()
                    
                    if text and text.lower() != ".":  # Filter out empty or punctuation-only results
                        all_texts.append(text)
                        lang_short = normalize_azure_lang(lang_result)
                        save_line_transcript(text, lang_short, i + 1, "audio_file")
                        print(f" Chunk {i+1} ({lang_short}): {text}")
                    else:
                        print(f" Chunk {i+1}: No meaningful speech detected")
                        
                elif result.reason == speechsdk.ResultReason.NoMatch:
                    print(f" No speech detected in chunk {i+1}")
                elif result.reason == speechsdk.ResultReason.Canceled:
                    cancellation = result.cancellation_details
                    print(f" Azure canceled chunk {i+1}: {cancellation.reason}")
                    if cancellation.reason == speechsdk.CancellationReason.Error:
                        print(f"   Error details: {cancellation.error_details}")

                # Clean up temporary chunk file
                try:
                    if os.path.exists(chunk_path):
                        os.unlink(chunk_path)
                except:
                    pass
                    
                time.sleep(0.3)  # Reduced delay

            except Exception as e:
                print(f" Error processing chunk {i+1}: {e}")
                # Continue with next chunk

        # Process results
        if all_texts:
            full_text = " ".join(all_texts)
            lang = normalize_azure_lang(detected_lang) if detected_lang else "en"
            save_final_transcript(all_texts, lang, "audio_file")
            print(f" Transcription completed. Detected language: {lang}")
            print(f" Full text: {full_text[:200]}...")
            return full_text, lang
        else:
            print(f" No speech detected in file: {original_file_path}")
            return "", "en-US"

    def recognize_from_microphone(self):
        """Recognize speech from microphone with IMPROVED handling"""
        print(" Listening... Speak clearly (English, Hindi, Spanish, French supported)")
        
        try:
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                auto_detect_source_language_config=self.auto_detect_config,
                audio_config=audio_config
            )

            print(" Recording... (speak now)")
            result = recognizer.recognize_once_async().get()
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                detected_lang = result.properties.get(
                    speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult, "en-US"
                )
                text = result.text.strip()
                if text and text.lower() != ".":
                    lang_short = normalize_azure_lang(detected_lang)
                    print(f" Recognized ({lang_short}): {text}")
                    save_line_transcript(text, lang_short, 1, "microphone")
                    return text, detected_lang
                else:
                    print(" No speech detected or empty result")
            elif result.reason == speechsdk.ResultReason.NoMatch:
                print(" No recognizable speech input.")
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancel = result.cancellation_details
                print(f" Canceled: {cancel.reason}")
                if cancel.error_details:
                    print(f"   Details: {cancel.error_details}")
                    
        except Exception as e:
            print(f" Microphone recognition error: {e}")
            
        return "", "en-US"

# Compatibility wrapper
def recognize_from_mic():
    try:
        return EnhancedSpeechRecognizer().recognize_from_microphone()
    except Exception as e:
        print(f" Speech recognizer failed: {e}")
        return "", "en-US"