# db_json.py 

import os
import json
import time
from datetime import datetime

# ----------------- Absolute Directories - USING YOUR EXISTING STRUCTURE -----------------

# Get the current project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

TRANSCRIPT_FILES = {
    "en-US": os.path.join(PROJECT_DIR, "results", "transcriptions", "english_transcriptions.json"),
    "hi-IN": os.path.join(PROJECT_DIR, "results", "transcriptions", "hindi_transcriptions.json"),
}

ENGLISH_TRANSLATIONS_DIR = os.path.join(PROJECT_DIR, "results", "translations", "english_translations")
HINDI_TRANSLATIONS_DIR = os.path.join(PROJECT_DIR, "results", "translations", "hindi_translations")

print(f"🔍 DEBUG: Using RELATIVE transcription files:")
print(f"  English: {TRANSCRIPT_FILES['en-US']}")
print(f"  Hindi: {TRANSCRIPT_FILES['hi-IN']}")
print(f"🔍 DEBUG: Using RELATIVE translation directories:")
print(f"  English: {ENGLISH_TRANSLATIONS_DIR}")
print(f"  Hindi: {HINDI_TRANSLATIONS_DIR}")

# ----------------- JSON Helpers - ENHANCED FOR APPENDING -----------------
def _load_json(path):
    """Load JSON data from file - PRESERVES EXISTING DATA"""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    print(f"📄 File is empty: {path}")
                    return {}
                data = json.loads(content)
                # Ensure we always return a dict
                if isinstance(data, list):
                    print(f"🔄 Converting list to dict for: {path}")
                    return {f"item_{i}": item for i, item in enumerate(data)}
                print(f"📖 Loaded existing data from {path}: {len(data)} entries")
                return data
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON decode error for {path}: {e}")
            return {}
        except Exception as e:
            print(f"⚠️  Error loading {path}: {e}")
            return {}
    else:
        print(f"📄 Creating new file: {path}")
        return {}

def _save_json(path, data):
    """Save data to JSON file - PRESERVES ALL DATA"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Write directly to preserve all data
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Verify the file was written
        if os.path.exists(path):
            file_size = os.path.getsize(path)
            entries_count = len(data) if isinstance(data, dict) else 0
            print(f"💾 Saved: {os.path.basename(path)} ({file_size} bytes, {entries_count} entries)")
            return True
        else:
            print(f"❌ File was not created: {path}")
            return False
            
    except Exception as e:
        print(f"❌ Error saving to {path}: {e}")
        return False

# ----------------- UNIFIED TRANSCRIPT SAVING - ALL SOURCES USE SAME FILES -----------------
def save_line_transcript(original_text, detected_language, line_num, source_type="microphone"):
    """Save individual line - APPENDS TO EXISTING DATA for ALL sources"""
    try:
        print(f"\n📝 SAVING TRANSCRIPT - Line {line_num} [{source_type.upper()}]")
        print(f"   Language: {detected_language}")
        print(f"   Text: {original_text[:80]}{'...' if len(original_text) > 80 else ''}")
        
        # Determine which transcription file to use
        if detected_language == 'hi-IN' or 'hi' in str(detected_language).lower():
            transcript_file = TRANSCRIPT_FILES["hi-IN"]
            lang_name = "Hindi"
        else:
            transcript_file = TRANSCRIPT_FILES["en-US"]
            lang_name = "English"
        
        print(f"   📁 Using EXISTING {lang_name} file")
        
        # Load EXISTING data
        data = _load_json(transcript_file)
        if not isinstance(data, dict):
            print(f"   ⚠️  Initializing new dict for existing file")
            data = {}
        
        previous_count = len(data)
        print(f"   📊 Existing entries: {previous_count}")
        
        # Create unique line ID to avoid overwrites
        timestamp = int(time.time())
        line_id = f"{source_type}_line_{line_num}_{timestamp}"
        
        # Add new entry to EXISTING data
        data[line_id] = {
            "timestamp": datetime.now().isoformat(),
            "line_number": line_num,
            "language": detected_language,
            "text": original_text,
            "source": source_type,
            "source_type": source_type
        }
        
        print(f"   ➕ Added to existing data: {line_id}")
        
        # Save back to EXISTING file
        if _save_json(transcript_file, data):
            new_count = len(data)
            print(f"   ✅ SUCCESS: Added to {lang_name} transcripts")
            print(f"   📈 Entries: {previous_count} → {new_count}")
            return True
        else:
            print(f"   ❌ FAILED: Could not save to existing file")
            return False
        
    except Exception as e:
        print(f"   ❌ ERROR saving transcript: {e}")
        return False

def save_final_transcript(lines, detected_language, source_type="audio_file"):
    """Save final complete transcript - APPENDS TO EXISTING for ALL sources"""
    try:
        complete_text = " ".join(lines)
        print(f"\n💾 SAVING FINAL TRANSCRIPT [{source_type.upper()}]")
        print(f"   Total lines: {len(lines)}")
        print(f"   Detected language: {detected_language}")
        
        # Determine which transcription file to use
        if detected_language == 'hi-IN' or 'hi' in detected_language.lower():
            transcript_file = TRANSCRIPT_FILES["hi-IN"]
            lang_name = "Hindi"
        else:
            transcript_file = TRANSCRIPT_FILES["en-US"]
            lang_name = "English"
        
        print(f"   📁 Using {lang_name} file: {transcript_file}")
        
        # Load existing data
        data = _load_json(transcript_file)
        print(f"   📊 Current file has {len(data)} entries")
        
        # Add complete transcript entry
        timestamp = int(time.time())
        complete_id = f"{source_type}_complete_{timestamp}"
        
        data[complete_id] = {
            "timestamp": datetime.now().isoformat(),
            "language": detected_language,
            "text": complete_text,
            "total_lines": len(lines),
            "source": source_type,
            "source_type": source_type,
            "is_complete": True
        }
        
        print(f"   ➕ Added complete transcript entry")
        
        # Save back
        if _save_json(transcript_file, data):
            # Verify file was written
            if os.path.exists(transcript_file):
                file_size = os.path.getsize(transcript_file)
                print(f"   ✅ SUCCESS: Complete transcript saved!")
                print(f"   📁 File: {os.path.basename(transcript_file)}")
                print(f"   📏 Size: {file_size} bytes")
                return complete_text
            else:
                print(f"   ❌ FAILED: File was not created")
                return None
        else:
            print(f"   ❌ FAILED: Could not save complete transcript")
            return None
            
    except Exception as e:
        print(f"   ❌ ERROR saving final transcript: {e}")
        import traceback
        traceback.print_exc()
        return None

# ----------------- UNIFIED TRANSLATION SAVING - ALL SOURCES USE SAME FILES -----------------
def save_translation(original_text, translated_text, source_lang, target_lang, source_type="microphone", line_num=1):
    """Save translation - APPENDS TO EXISTING FILES for ALL sources"""
    try:
        print(f"\n🌐 SAVING TRANSLATION [{source_type.upper()}] - Line {line_num}")
        print(f"   Source: {source_lang} → Target: {target_lang}")
        print(f"   Original: {original_text[:60]}...")
        print(f"   Translated: {translated_text[:60]}...")
        
        # Determine which translation directory to use
        if source_lang == 'hi-IN' or 'hi' in source_lang.lower():
            translation_dir = HINDI_TRANSLATIONS_DIR
            source_name = "Hindi"
        else:
            translation_dir = ENGLISH_TRANSLATIONS_DIR
            source_name = "English"
        
        print(f"   📁 Using EXISTING {source_name} translation directory")
        
        # Use EXISTING translation file
        translation_file = os.path.join(translation_dir, f"{target_lang}_translations.json")
        
        # Load EXISTING translations
        all_translations = _load_json(translation_file)
        if not isinstance(all_translations, dict):
            print(f"   ⚠️  Initializing dict for existing file")
            all_translations = {}
        
        previous_entries = len(all_translations)
        
        # Create unique ID
        timestamp = int(time.time())
        translation_id = f"{source_type}_line_{line_num}_{timestamp}"
        
        # Add to EXISTING data
        all_translations[translation_id] = {
            'original_text': original_text,
            'translated_text': translated_text,
            'timestamp': datetime.now().isoformat(),
            'source_language': source_lang,
            'target_language': target_lang,
            'line_number': line_num,
            'source': source_type,
            'source_type': source_type,
            'translation_id': translation_id
        }
        
        print(f"   ➕ Adding translation to {target_lang} file")
        
        # Save back to EXISTING file
        if _save_json(translation_file, all_translations):
            new_entries = len(all_translations)
            print(f"   ✅ {target_lang.upper()}: {previous_entries} → {new_entries} entries")
            return True
        else:
            print(f"   ❌ Failed to save {target_lang} translation")
            return False
        
    except Exception as e:
        print(f"   ❌ Error saving translation: {e}")
        return False

def save_line_translation(original_text, source_lang, translations, line_num, source_type="microphone"):
    """Save multiple translations - APPENDS TO EXISTING FILES for ALL sources"""
    try:
        print(f"\n🌐 SAVING TRANSLATIONS [{source_type.upper()}] - Line {line_num}")
        print(f"   Source: {source_lang}")
        print(f"   Translating to: {list(translations.keys())}")
        
        # Determine which translation directory to use
        if source_lang == 'hi-IN' or 'hi' in source_lang.lower():
            translation_dir = HINDI_TRANSLATIONS_DIR
            source_name = "Hindi"
        else:
            translation_dir = ENGLISH_TRANSLATIONS_DIR
            source_name = "English"
        
        print(f"   📁 Using EXISTING {source_name} translation directory")
        
        # Save to EXISTING language files
        saved_count = 0
        for target_lang, translated_text in translations.items():
            try:
                # Save each translation using the unified function
                success = save_translation(
                    original_text, 
                    translated_text, 
                    source_lang, 
                    target_lang, 
                    source_type, 
                    line_num
                )
                
                if success:
                    saved_count += 1
                else:
                    print(f"      ❌ Failed to save {target_lang} translation")
                    
            except Exception as e:
                print(f"      ❌ Error saving {target_lang}: {e}")
                continue
        
        print(f"   ✅ Line {line_num}: {saved_count}/{len(translations)} translations saved")
        return saved_count > 0
        
    except Exception as e:
        print(f"   ❌ Error saving translations: {e}")
        return False

# ----------------- BACKWARD COMPATIBILITY FUNCTIONS -----------------
def save_language_specific_translation(source_lang, target_lang, original_text, translated_text, sentence_num=None, source_type="audio_file"):
    """Backward compatibility function - Save translation to language-specific file"""
    return save_translation(
        original_text, 
        translated_text, 
        source_lang, 
        target_lang, 
        source_type, 
        sentence_num or 1
    )

# ----------------- STATISTICS FUNCTIONS -----------------
def get_translation_stats():
    """Get statistics about translations"""
    stats = {
        'total_translations': 0,
        'by_language': {},
        'by_source': {},
        'recent_activity': []
    }

    # Count translations in language-specific directories
    for source_lang, source_dir in [('en', ENGLISH_TRANSLATIONS_DIR), ('hi', HINDI_TRANSLATIONS_DIR)]:
        if os.path.exists(source_dir):
            for file in os.listdir(source_dir):
                if file.endswith('_translations.json'):
                    file_path = os.path.join(source_dir, file)
                    data = _load_json(file_path)
                    if data and isinstance(data, dict):
                        target_lang = file.replace('_translations.json', '')
                        stats['by_language'][f"{source_lang}-{target_lang}"] = len(data)
                        stats['total_translations'] += len(data)
                        
                        # Count by source type
                        for entry in data.values():
                            source_type = entry.get('source_type', 'unknown')
                            stats['by_source'][source_type] = stats['by_source'].get(source_type, 0) + 1

    return stats

def get_language_specific_stats():
    """Get statistics for language-specific translations"""
    stats = {}
    
    for source_lang, source_dir in [('en', ENGLISH_TRANSLATIONS_DIR), ('hi', HINDI_TRANSLATIONS_DIR)]:
        if not os.path.exists(source_dir):
            continue
            
        for file in os.listdir(source_dir):
            if file.endswith('_translations.json'):
                lang_code = file.replace('_translations.json', '')
                file_path = os.path.join(source_dir, file)
                data = _load_json(file_path)
                if data and isinstance(data, dict):
                    stats[f"{source_lang}-{lang_code}"] = len(data)
    
    return stats

def get_source_statistics():
    """Get statistics by source type"""
    source_stats = {}
    
    # Check transcription files
    for lang, file_path in TRANSCRIPT_FILES.items():
        data = _load_json(file_path)
        if data and isinstance(data, dict):
            for entry_id, entry in data.items():
                source_type = entry.get('source_type', 'unknown')
                source_stats[source_type] = source_stats.get(source_type, 0) + 1
    
    return source_stats

# ----------------- FILE CHECKING & DEBUGGING -----------------
def check_transcription_files():
    """Debug function to check if transcription files are being updated"""
    print("\n" + "="*60)
    print("🔍 CHECKING TRANSCRIPTION FILES - DETAILED DEBUG")
    print("="*60)
    
    for lang, file_path in TRANSCRIPT_FILES.items():
        lang_name = "Hindi" if "hindi" in file_path.lower() else "English"
        print(f"\n📄 {lang_name} Transcript File:")
        print(f"   Path: {file_path}")
        
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"   ✅ File exists: {file_size} bytes")
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    print(f"   📖 File content length: {len(content)} characters")
                    
                data = _load_json(file_path)
                if isinstance(data, dict):
                    line_count = len(data)
                    print(f"   📊 Total entries: {line_count}")
                    
                    # Show source statistics
                    source_counts = {}
                    for entry in data.values():
                        source_type = entry.get('source_type', 'unknown')
                        source_counts[source_type] = source_counts.get(source_type, 0) + 1
                    
                    if source_counts:
                        print(f"   📈 Source distribution:")
                        for source, count in source_counts.items():
                            print(f"      - {source}: {count} entries")
                    
                    # Show recent entries
                    if data:
                        print(f"   📋 Recent entries:")
                        keys = list(data.keys())[-3:]  # Last 3 entries
                        for key in keys:
                            value = data[key]
                            text_preview = value.get('text', '')[:50] + "..." if len(value.get('text', '')) > 50 else value.get('text', '')
                            source_type = value.get('source_type', 'unknown')
                            print(f"     - {key} [{source_type}]: {text_preview}")
                    else:
                        print(f"   📭 File is empty")
                else:
                    print(f"   ⚠️  File content is not a dictionary: {type(data)}")
                    
            except Exception as e:
                print(f"   ❌ Error reading file: {e}")
        else:
            print(f"   ❌ File does not exist!")
    
    # Also check translation directories
    print("\n" + "="*60)
    print("🔍 CHECKING TRANSLATION DIRECTORIES")
    print("="*60)
    
    translation_dirs = [
        (ENGLISH_TRANSLATIONS_DIR, "English translations"),
        (HINDI_TRANSLATIONS_DIR, "Hindi translations")
    ]
    
    for dir_path, dir_name in translation_dirs:
        print(f"\n📁 {dir_name}:")
        print(f"   Path: {dir_path}")
        
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
            print(f"   ✅ Directory exists: {len(files)} translation files")
            
            for file in files:
                file_path = os.path.join(dir_path, file)
                file_size = os.path.getsize(file_path)
                try:
                    data = _load_json(file_path)
                    entry_count = len(data) if isinstance(data, dict) else "unknown"
                    
                    # Get source statistics
                    source_counts = {}
                    if data and isinstance(data, dict):
                        for entry in data.values():
                            source_type = entry.get('source_type', 'unknown')
                            source_counts[source_type] = source_counts.get(source_type, 0) + 1
                    
                    print(f"   📄 {file}: {file_size} bytes, {entry_count} entries")
                    if source_counts:
                        sources_str = ", ".join([f"{k}:{v}" for k, v in source_counts.items()])
                        print(f"        📊 Sources: {sources_str}")
                except:
                    print(f"   📄 {file}: {file_size} bytes")
        else:
            print(f"   ❌ Directory does not exist!")

# ----------------- ENHANCED DEBUG FUNCTION -----------------
def check_all_existing_files():
    """Check ALL existing files in your structure"""
    print("\n" + "="*80)
    print("🔍 CHECKING ALL EXISTING FILES IN YOUR STRUCTURE")
    print("="*80)
    
    # Check all directories that should exist
    directories_to_check = [
        ("Transcriptions", os.path.dirname(TRANSCRIPT_FILES["en-US"])),
        ("English Translations", ENGLISH_TRANSLATIONS_DIR),
        ("Hindi Translations", HINDI_TRANSLATIONS_DIR),
    ]
    
    for dir_name, dir_path in directories_to_check:
        print(f"\n📁 {dir_name}:")
        print(f"   Path: {dir_path}")
        
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
            print(f"   ✅ Directory exists: {len(files)} files")
            
            for file in sorted(files):
                file_path = os.path.join(dir_path, file)
                data = _load_json(file_path)
                file_size = os.path.getsize(file_path)
                entries = len(data) if isinstance(data, dict) else 0
                
                # Get source statistics for this file
                source_counts = {}
                if data and isinstance(data, dict):
                    for entry in data.values():
                        source_type = entry.get('source_type', 'unknown')
                        source_counts[source_type] = source_counts.get(source_type, 0) + 1
                
                print(f"   📄 {file}: {file_size:6} bytes, {entries:3} entries")
                if source_counts:
                    sources_str = ", ".join([f"{k}:{v}" for k, v in source_counts.items()])
                    print(f"        📊 Sources: {sources_str}")
                    
                # Show sample of recent entries
                if entries > 0:
                    keys = list(data.keys())[-2:]  # Last 2 entries
                    for key in keys:
                        if 'text' in data[key]:
                            text = data[key]['text'][:50] + "..." if len(data[key]['text']) > 50 else data[key]['text']
                            source_type = data[key].get('source_type', 'unknown')
                            print(f"        📝 {key} [{source_type}]: {text}")
        else:
            print(f"   ❌ Directory missing!")

# ----------------- UTILITY FUNCTIONS -----------------
def get_recent_transcriptions(limit=10, source_type=None):
    """Get recent transcriptions with optional source filter"""
    all_transcriptions = []

    for lang, file_path in TRANSCRIPT_FILES.items():
        data = _load_json(file_path)
        if data and isinstance(data, dict):
            # Convert dict to list of entries
            entries = []
            for key, value in data.items():
                if key != "complete_transcript":
                    value['entry_id'] = key
                    entries.append(value)
            
            # Filter by source type if specified
            if source_type:
                entries = [e for e in entries if e.get('source_type') == source_type]
            
            # Sort by timestamp and get recent ones
            entries.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            for entry in entries[:limit]:
                entry['source_file'] = os.path.basename(file_path)
                all_transcriptions.append(entry)

    return all_transcriptions

def get_recent_translations(limit=10, source_type=None):
    """Get recent translations with optional source filter"""
    recent_translations = []
    
    for source_lang, source_dir in [('en', ENGLISH_TRANSLATIONS_DIR), ('hi', HINDI_TRANSLATIONS_DIR)]:
        if not os.path.exists(source_dir):
            continue
            
        for file in os.listdir(source_dir):
            if file.endswith('_translations.json'):
                file_path = os.path.join(source_dir, file)
                data = _load_json(file_path)
                if data and isinstance(data, dict):
                    # Convert to list and filter
                    entries = list(data.values())
                    
                    # Filter by source type if specified
                    if source_type:
                        entries = [e for e in entries if e.get('source_type') == source_type]
                    
                    # Sort by timestamp and get recent ones
                    entries.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                    for entry in entries[:limit]:
                        recent_translations.append(entry)
    
    return recent_translations[:limit]

# ----------------- TEST WITH EXISTING FILES -----------------
def test_with_existing_files():
    """Test that we're properly using existing files"""
    print("\n" + "="*80)
    print("🧪 TESTING WITH EXISTING FILES")
    print("="*80)
    
    # First, show current state
    check_all_existing_files()
    
    # Test adding new data with different source types
    test_text = "This is a test to verify we append to existing files with proper source types"
    test_translations = {'hi': 'यह एक परीक्षण है', 'es': 'Esta es una prueba'}
    
    print(f"\n➕ ADDING TEST DATA TO EXISTING FILES:")
    
    # Add transcripts with different source types
    source_types = ["youtube", "microphone", "audio_file", "online_audio"]
    
    for i, source_type in enumerate(source_types, 1):
        print(f"\n--- Testing {source_type.upper()} ---")
        # Add transcript
        transcript_result = save_line_transcript(f"{test_text} - {source_type}", "en-US", 9000 + i, source_type)
        
        # Add translations  
        translation_result = save_line_translation(
            f"{test_text} - {source_type}", 
            "en-US", 
            test_translations, 
            9000 + i, 
            source_type
        )
    
    # Show final state
    print(f"\n📊 FINAL STATE AFTER ADDING TEST DATA:")
    check_all_existing_files()
    
    # Show source statistics
    source_stats = get_source_statistics()
    print(f"\n📈 SOURCE STATISTICS:")
    for source_type, count in source_stats.items():
        print(f"   {source_type}: {count} entries")
    
    return True

# Run the test when file is executed directly
if __name__ == "__main__":
    print("🚀 TESTING UNIFIED FILE STRUCTURE...")
    test_with_existing_files()
