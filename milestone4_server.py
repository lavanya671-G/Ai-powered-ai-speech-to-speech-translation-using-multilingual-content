# milestone4_server.py
# ------------------------------------------
# Flask + SocketIO backend for Milestone4
# Improved: chunked STT (long-form), robust TTS & merge, progress events
# ------------------------------------------

import os
os.environ["PATH"] += os.pathsep + r"D:\Projects\live-ai-speech-translator\ffmpeg\bin"

import io
import uuid
import json
import base64
import shutil
import subprocess
import threading
import time
from datetime import datetime
from functools import wraps

from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_socketio import SocketIO, emit
import requests
from dotenv import load_dotenv
load_dotenv()

# Optional audio libs
try:
    import azure.cognitiveservices.speech as speechsdk
except Exception:
    speechsdk = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    from pydub import AudioSegment
    # Ensure pydub knows ffmpeg locations on windows if needed
    AudioSegment.converter = r"D:\Projects\live-ai-speech-translator\ffmpeg\bin\ffmpeg.exe"
    AudioSegment.ffprobe = r"D:\Projects\live-ai-speech-translator\ffmpeg\bin\ffprobe.exe"
except Exception:
    AudioSegment = None

# CONFIG
TRANSLATOR_KEY = os.getenv("TRANSLATOR_KEY")
TRANSLATOR_REGION = os.getenv("TRANSLATOR_REGION", "centralindia")
TRANSLATOR_ENDPOINT = os.getenv("TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com/")

SPEECH_KEY = os.getenv("SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION", "centralindia")

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

FFMPEG_BIN = r"D:\Projects\live-ai-speech-translator\ffmpeg\bin\ffmpeg.exe"
FFPROBE_BIN = r"D:\Projects\live-ai-speech-translator\ffmpeg\bin\ffprobe.exe"

# Voice map: short code -> Azure voice name (adjust to available voices)
VOICE_MAP = {
    "hi": "hi-IN-SwaraNeural",
    "en": "en-US-Neural2-A",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "it": "it-IT-ElsaNeural",
    "ru": "ru-RU-DariyaNeural",
    "ar": "ar-SA-ZariyahNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "nl": "nl-NL-AnnaNeural",
    "pt": "pt-PT-HeloisaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural"
}

# Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET", "dev-secret")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Ensure folders
os.makedirs("static/uploads/videos/original", exist_ok=True)
os.makedirs("static/uploads/videos/translated", exist_ok=True)
os.makedirs("results/audio_output", exist_ok=True)
os.makedirs("results/temp_audio", exist_ok=True)
os.makedirs("results/translations", exist_ok=True)

# ---------------------------
# Utilities
# ---------------------------
def safe_run(cmd_list, timeout=None):
    try:
        proc = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=timeout)
        return True, proc.stdout.decode(errors='ignore') + proc.stderr.decode(errors='ignore')
    except subprocess.CalledProcessError as e:
        return False, (e.stdout.decode(errors='ignore') + e.stderr.decode(errors='ignore'))
    except Exception as e:
        return False, str(e)

def ffmpeg_extract_audio(video_path, out_wav):
    cmd = [FFMPEG_BIN, "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", out_wav]
    ok, out = safe_run(cmd, timeout=300)
    if not ok:
        print("ffmpeg_extract_audio error:", out)
    return out_wav if ok else None

def ffmpeg_merge_audio_into_video(video_path, audio_path, out_video):
    # Merge audio (re-encode audio to aac if needed) and copy video stream
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        out_video
    ]
    ok, out = safe_run(cmd, timeout=300)
    if not ok:
        print("FFmpeg merge failed:", out)
        return None
    return out_video

def save_audio_from_base64(b64string, out_dir="results/temp_audio"):
    """legacy helper - decodes base64 and converts to wav (used internally by wrapper)"""
    os.makedirs(out_dir, exist_ok=True)
    try:
        if b64string.startswith("data:"):
            b64string = b64string.split(",", 1)[1]
        raw = base64.b64decode(b64string)
        fid = uuid.uuid4().hex
        webm_path = os.path.join(out_dir, f"{fid}.webm")
        wav_path = os.path.join(out_dir, f"{fid}.wav")
        with open(webm_path, "wb") as f:
            f.write(raw)
        ok, out = safe_run([FFMPEG_BIN, "-y", "-i", webm_path, "-ac", "1", "-ar", "16000", wav_path], timeout=60)
        if not ok:
            print("ffmpeg conversion error:", out)
            return None
        return os.path.abspath(wav_path)
    except Exception as e:
        print("save_audio_from_base64 error:", e)
        return None

def save_temp_audio(b64_audio):
    """
    General wrapper used by mic-translate handlers.
    Returns absolute WAV path or None.
    """
    try:
        if not b64_audio:
            print("save_temp_audio: empty payload")
            return None
        # defensive: sometimes frontend sends object
        if isinstance(b64_audio, dict):
            b64_audio = b64_audio.get("audio_b64") or b64_audio.get("data") or ""
        wav = save_audio_from_base64(b64_audio, out_dir="results/temp_audio")
        if wav:
            return wav
        return None
    except Exception as e:
        print("save_temp_audio error:", e)
        return None

# ---------------------------
# Azure Services
# ---------------------------
def azure_stt_from_wav_once(wav_path, language="en-US"):
    # short utterance STT (keeps mic flow working)
    if speechsdk is None:
        print("Azure Speech SDK not installed.")
        return "", language
    if not SPEECH_KEY or not SPEECH_REGION:
        print("Azure speech key/region not set.")
        return "", language

    wav_path = os.path.abspath(wav_path)
    try:
        cfg = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
        cfg.speech_recognition_language = language
        audio = speechsdk.audio.AudioConfig(filename=wav_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio)
        result = recognizer.recognize_once()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text, language
        else:
            return "", language
    except Exception as e:
        print("azure_stt_from_wav_once error:", e)
        return "", language

# --- NEW: chunking helpers for long audio transcription ---
def split_audio_into_chunks(wav_path, chunk_length_ms=15000, out_dir=None):
    """
    Split WAV into chunks (pydub). chunk_length_ms default 15s.
    Returns list of absolute chunk paths.
    """
    if AudioSegment is None:
        print("pydub not available; cannot split.")
        return []

    if out_dir is None:
        out_dir = os.path.dirname(wav_path)
    os.makedirs(out_dir, exist_ok=True)

    audio = AudioSegment.from_wav(wav_path)
    total_ms = len(audio)
    chunks = []
    idx = 0
    for start in range(0, total_ms, chunk_length_ms):
        end = min(start + chunk_length_ms, total_ms)
        part = audio[start:end]
        chunk_name = f"{os.path.splitext(os.path.basename(wav_path))[0]}_chunk_{idx}.wav"
        chunk_path = os.path.join(out_dir, chunk_name)
        part.export(chunk_path, format="wav")
        chunks.append(os.path.abspath(chunk_path))
        idx += 1
    return chunks

def azure_stt_long_form(wav_path, language="en-US", chunk_ms=20000, sid=None):
    """
    Transcribe long wav by splitting into chunks and calling recognize_once per chunk.
    chunk_ms default 20s (tweak to balance accuracy/time).
    Returns combined transcription string.
    Emits progress events to socket if sid provided.
    """
    if speechsdk is None:
        print("Azure Speech SDK not installed.")
        return ""

    wav_abs = os.path.abspath(wav_path)
    if not os.path.exists(wav_abs):
        print("wav missing:", wav_abs)
        return ""

    # create chunks
    try:
        if sid:
            socketio.emit("translate-progress", {"status": "stt_splitting", "detail": "splitting audio into chunks"}, room=sid)
        chunks = split_audio_into_chunks(wav_abs, chunk_length_ms=chunk_ms, out_dir=os.path.dirname(wav_abs))
        if not chunks:
            print("No chunks produced.")
            return ""
    except Exception as e:
        print("split error:", e)
        return ""

    # STT per chunk
    combined = []
    cfg = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    cfg.speech_recognition_language = language

    for i, chunk in enumerate(chunks):
        try:
            if sid:
                socketio.emit("translate-progress", {"status": "stt_chunk", "detail": f"processing chunk {i+1}/{len(chunks)}"}, room=sid)
            audio_cfg = speechsdk.audio.AudioConfig(filename=chunk)
            recognizer = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio_cfg)
            result = recognizer.recognize_once()
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                text = result.text.strip()
                combined.append(text)
            else:
                print(f"Chunk {i} result reason:", result.reason)
                combined.append("")
        except Exception as e:
            print("azure_stt_long_form chunk error:", e)
            combined.append("")
        finally:
            try:
                os.remove(chunk)
            except Exception:
                pass

    final_text = " ".join([t for t in combined if t]).strip()
    return final_text

def azure_translate_text(text, target_lang="hi"):
    if not TRANSLATOR_KEY:
        return None, {"error": "Missing TRANSLATOR_KEY"}
    headers = {
        "Ocp-Apim-Subscription-Key": TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": TRANSLATOR_REGION,
        "Content-Type": "application/json"
    }
    params = {"api-version": "3.0", "to": target_lang}
    body = [{"text": text}]
    try:
        url = f"{TRANSLATOR_ENDPOINT.rstrip('/')}/translate"
        resp = requests.post(url, params=params, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data[0]["translations"][0]["text"], data
    except Exception as e:
        print("azure_translate_text error:", e)
        return None, {"error": str(e)}

def azure_tts_save_wav(text, language_code="hi", voice_name=None):
    """
    Robust TTS:
     - Tries WAV PCM first (Riff16Khz16BitMonoPcm)
     - Then tries MP3 (Audio16Khz32KBitRateMonoMp3)
     - Falls back to pyttsx3 if Azure not available
    Returns (web_url, disk_path_or_error)
    """
    out_dir = "results/audio_output"
    os.makedirs(out_dir, exist_ok=True)

    # Try a list of formats (format constant, extension)
    formats_to_try = []
    if speechsdk:
        # prefer WAV PCM (widely supported) then MP3
        try:
            formats_to_try.append((speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm, "wav"))
        except Exception:
            pass
        try:
            formats_to_try.append((speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3, "mp3"))
        except Exception:
            pass

    # If nothing from sdk (or not available), fallback to mp3 extension for pyttsx3
    if not formats_to_try:
        formats_to_try.append((None, "wav"))

    last_err = None
    # If Azure SDK available and keys set, try Azure synthesis
    if speechsdk and SPEECH_KEY and SPEECH_REGION:
        for fmt_const, ext in formats_to_try:
            filename = f"tts_{uuid.uuid4().hex}.{ext}"
            out_path = os.path.join(out_dir, filename)
            try:
                cfg = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
                short = (language_code[:2].lower() if language_code else "en")
                voice = voice_name or VOICE_MAP.get(short, None)
                if voice:
                    cfg.speech_synthesis_voice_name = voice
                else:
                    cfg.speech_synthesis_voice_name = f"{short}-Neural-A" if isinstance(short, str) else "en-US-Neural2-A"

                if fmt_const is not None:
                    try:
                        cfg.set_speech_synthesis_output_format(fmt_const)
                    except Exception as e:
                        # continue to next format if setting format fails
                        print("Could not set format on cfg:", e)

                audio_cfg = speechsdk.audio.AudioOutputConfig(filename=out_path)
                synthesizer = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=audio_cfg)
                result = synthesizer.speak_text_async(text).get()

                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    return f"/audio/{filename}", out_path
                else:
                    err = getattr(result, "error_details", None) or "unknown tts reason"
                    last_err = f"TTS failed (format {ext}): {err}"
                    print(last_err)
                    # try next format
            except Exception as e:
                last_err = f"TTS exception (format {ext}): {e}"
                print(last_err)
                # try next format

    # Fallback pyttsx3 if Azure failed or unavailable
    if pyttsx3:
        try:
            filename = f"tts_{uuid.uuid4().hex}.wav"
            out_path = os.path.join(out_dir, filename)
            engine = pyttsx3.init()
            engine.save_to_file(text, out_path)
            engine.runAndWait()
            return f"/audio/{filename}", out_path
        except Exception as e:
            last_err = f"pyttsx3 failed: {e}"
            print(last_err)

    return None, last_err or "No TTS method succeeded"

# ---------------------------
# Helpers (semantic)
# ---------------------------
def semantic_cleanup(text):
    try:
        if not text or not isinstance(text, str):
            return ""
        cleaned = text.strip()
        cleaned = cleaned.replace("\n", " ").strip()
        # small normalization heuristics
        cleaned = " ".join(cleaned.split())
        return cleaned
    except Exception as e:
        print("semantic_cleanup error:", e)
        return text or ""

# ---------------------------
# Flask Routes
# ---------------------------
@app.route("/videos")
def list_videos():
    original_dir = "static/uploads/videos/original"
    translated_dir = "static/uploads/videos/translated"
    COMMON_LANGS = ["hi","en","fr","es","de","it","ru","ar","zh","nl","pt","ja","ko"]

    originals = []
    translated = []

    if os.path.exists(original_dir):
        for f in os.listdir(original_dir):
            if f.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
                originals.append({
                    "filename": f,
                    "src": f"/static/uploads/videos/original/{f}",
                    "title": f,
                    "available_langs": COMMON_LANGS
                })

    if os.path.exists(translated_dir):
        for f in os.listdir(translated_dir):
            if f.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
                translated.append({
                    "filename": f,
                    "src": f"/static/uploads/videos/translated/{f}",
                    "title": f
                })

    return jsonify({
        "original_videos": originals,
        "translated_videos": translated
    })

# mic-translate (keeps previous behavior)
@socketio.on('mic-translate')
def handle_mic_translate(data):
    sid = request.sid
    try:
        socketio.emit("mic-translate-progress", {"status":"received"}, room=sid)
        audio_path = save_temp_audio(data.get("audio_b64") if isinstance(data, dict) else data)
        if not audio_path or not os.path.exists(audio_path):
            socketio.emit('mic-translate-result', {"success": False, "error": "audio save/convert failed"}, room=sid)
            return

        text, detected_lang = azure_stt_from_wav_once(audio_path)
        if not text:
            socketio.emit('mic-translate-result', {"success": False, "error": "speech not recognized"}, room=sid)
            return

        fixed_text = semantic_cleanup(text)
        target_lang = data.get("target_lang") or data.get("targetLanguage") or "hi"
        translated, raw = azure_translate_text(fixed_text, target_lang)
        if translated is None:
            socketio.emit('mic-translate-result', {"success": False, "error": "translation failed", "detail": raw}, room=sid)
            return

        audio_url, audio_disk_or_err = azure_tts_save_wav(translated, language_code=target_lang)
        if not audio_url:
            socketio.emit('mic-translate-result', {"success": False, "error": "tts failed", "detail": audio_disk_or_err}, room=sid)
            return

        socketio.emit("mic-translate-result", {
            "success": True,
            "original": fixed_text,
            "translated": translated,
            "audio_url": audio_url,
        }, room=sid)

    except Exception as e:
        print("mic pipeline error:", e)
        socketio.emit("mic-translate-result", {"success": False, "error": str(e)}, room=sid)

@app.route("/audio/<path:filename>")
def audio(filename):
    return send_from_directory("results/audio_output", filename)

@app.route("/translate-video", methods=["POST"])
def translate_video():
    data = request.get_json() or {}
    video_name = data.get("video_name")
    target_lang = (data.get("target_lang") or "").strip()
    sid = data.get("socket_sid")

    if not video_name:
        return jsonify({"error": "video_name required"}), 400
    if not target_lang:
        return jsonify({"error": "No target language selected"}), 400

    threading.Thread(target=background_translate_video_task, args=(video_name, target_lang, sid), daemon=True).start()
    return jsonify({"status": "accepted"})

# ---------------------------
# Background Task (main flow)
# ---------------------------
def background_translate_video_task(video_filename, target_lang, sid=None):
    try:
        if sid:
            socketio.emit("translate-progress", {"status":"accepted", "detail": f"Translating {video_filename} -> {target_lang}"}, room=sid)

        src = os.path.join("static", "uploads", "videos", "original", video_filename)
        if not os.path.exists(src):
            socketio.emit("translate-progress", {"status":"error", "message": f"Video not found: {src}"}, room=sid)
            return

        # extract audio
        if sid: socketio.emit("translate-progress", {"status":"extract_audio"}, room=sid)
        wav_out = os.path.join("results", "temp_audio", f"{uuid.uuid4().hex}.wav")
        wav = ffmpeg_extract_audio(src, wav_out)
        if not wav or not os.path.exists(wav):
            socketio.emit("translate-progress", {"status":"error", "message":"Audio extract failed"}, room=sid)
            return

        # STT - LONG FORM
        if sid: socketio.emit("translate-progress", {"status":"stt_start", "detail": "Starting speech recognition"}, room=sid)
        # For now we use default SR language; you can improve by auto-detecting source language
        sr_lang = "en-US"
        text = azure_stt_long_form(wav, language=sr_lang, chunk_ms=20000, sid=sid)
        if not text or len(text.strip()) < 3:
            socketio.emit("translate-progress", {"status":"error", "message": "Speech recognition failed / too short"}, room=sid)
            return

        if sid: socketio.emit("translate-progress", {"status":"stt_done", "detail": "Recognition complete"}, room=sid)

        cleaned = semantic_cleanup(text)
        if sid: socketio.emit("translate-progress", {"status":"translating", "detail": "Translating recognized text..."}, room=sid)
        translated_text, _ = azure_translate_text(cleaned, target_lang)
        if translated_text is None:
            socketio.emit("translate-progress", {"status":"error", "message":"Translation failed"}, room=sid)
            return

        if sid: socketio.emit("translate-progress", {"status":"tts_start", "detail":"Generating speech from translated text..."}, room=sid)
        audio_url, audio_disk_or_err = azure_tts_save_wav(translated_text, language_code=target_lang)
        if not audio_url or not os.path.exists(audio_disk_or_err):
            socketio.emit("translate-progress", {"status":"error", "message": f"TTS failed: {audio_disk_or_err}"}, room=sid)
            return

        if sid: socketio.emit("translate-progress", {"status":"merging", "detail":"Merging audio into video..."}, room=sid)
        out_video = os.path.join("static", "uploads", "videos", "translated", f"{os.path.splitext(video_filename)[0]}_translated_{target_lang}.mp4")
        merged = ffmpeg_merge_audio_into_video(src, audio_disk_or_err, out_video)
        if not merged or not os.path.exists(out_video):
            socketio.emit("translate-progress", {"status":"error", "message":"FFmpeg merging failed"}, room=sid)
            return

        if sid:
            socketio.emit("translate-progress", {"status":"done", "translated_video": "/" + out_video.replace("\\","/")}, room=sid)

    except Exception as e:
        print("background_translate_video_task error:", e)
        socketio.emit("translate-progress", {"status":"error", "message": str(e)}, room=sid)

@app.route("/delete-video", methods=["POST"])
def delete_video_api():
    data = request.get_json()
    filename = data.get("filename")
    video_type = data.get("type")  # "original" or "translated"

    if not filename or not video_type:
        return jsonify({"success": False, "message": "Missing filename/type"}), 400

    folder = "static/uploads/videos/original" if video_type == "original" \
             else "static/uploads/videos/translated"

    path = os.path.join(folder, filename)

    if os.path.exists(path):
        os.remove(path)
        return jsonify({"success": True})

    return jsonify({"success": False, "message": "File not found"}), 404

# Serve static videos (explicit)
@app.route('/static/uploads/videos/original/<path:filename>')
def serve_original_video(filename):
    return send_from_directory('static/uploads/videos/original', filename, as_attachment=False)

@app.route('/static/uploads/videos/translated/<path:filename>')
def serve_translated_video(filename):
    return send_from_directory('static/uploads/videos/translated', filename, as_attachment=False)

@app.route("/")
def index():
    return render_template("index.html")

# Run
if __name__ == "__main__":
    print("🚀 Milestone4 Flask Server running on http://127.0.0.1:5001")
    socketio.run(app, host="127.0.0.1", port=5001, debug=True)
