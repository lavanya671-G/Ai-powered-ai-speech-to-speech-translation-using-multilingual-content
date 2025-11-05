# utils_shared.py

"""
UTILS_SHARED.PY
Common helper functions and constants shared between app.py and translation_pipeline.py
"""

# ✅ Supported language codes
SUPPORTED_LANG_CODES = [
    'en', 'hi', 'es', 'fr', 'de', 'ja', 'zh', 'it', 'ru', 'ar',
     'ko', 'pt', 'zh', 'nl'
]

def ensure_str_from_translation(translation_output):
    """Ensure translation output is a clean string (handles dicts/lists returned by pipelines)."""
    if isinstance(translation_output, list):
        if len(translation_output) > 0 and isinstance(translation_output[0], dict):
            return translation_output[0].get('translation_text', str(translation_output[0]))
        else:
            return ' '.join(map(str, translation_output))
    elif isinstance(translation_output, dict):
        return translation_output.get('translation_text', str(translation_output))
    return str(translation_output)

def translate_text_single(translator_pipeline, text, target_lang):
    """Helper: Use Hugging Face translation pipeline for one target language."""
    try:
        result = translator_pipeline(text, max_length=512)
        return ensure_str_from_translation(result)
    except Exception as e:
        print(f"⚠️ Translation failed for {target_lang}: {e}")
        return None
