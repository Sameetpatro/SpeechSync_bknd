from flask import Flask, request, jsonify, send_file
import os
import whisper
import torch
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
from datetime import datetime
from werkzeug.utils import secure_filename
import traceback
import logging
import subprocess
import shlex

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Folders for file storage
UPLOAD_FOLDER = "uploads"
GENERATED_AUDIO_FOLDER = "generated_audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_AUDIO_FOLDER, exist_ok=True)

# Allowed audio extensions
ALLOWED_EXTENSIONS = {"mp3", "wav", "ogg", "m4a", "flac"}

# Global model variables (loaded once at startup)
whisper_model = None
translation_models = {}  # cache: key -> (model, tokenizer)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_whisper_model():
    """Load Whisper model once at startup"""
    global whisper_model
    try:
        logger.info("Loading Whisper model (base)...")
        # small/ base tradeoff: "base" is decent; change to "small" if you want better accuracy (bigger)
        whisper_model = whisper.load_model("small")
        logger.info("Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {str(e)}")
        raise

def get_translation_model(source_lang, target_lang):
    """
    Load translation model with fallback logic.
    Returns (model, tokenizer) tuple.
    Only en<->hi currently uses actual Helsinki models.
    Bengali/Odia are kept as placeholders (not implemented) to keep repo small.
    """
    global translation_models
    model_key = f"{source_lang}-{target_lang}"
    if model_key in translation_models:
        return translation_models[model_key]

    # Keep only the pairs we care about (en <-> hi). Others not implemented for now.
    model_mappings = {
        "en-hi": "Helsinki-NLP/opus-mt-en-hi",
        "hi-en": "Helsinki-NLP/opus-mt-hi-en",
        # Bengali/Odia placeholder entries (not downloaded by default)
        # "en-bn": "Helsinki-NLP/opus-mt-en-bn",  # <- not added to avoid unexpected downloads
        # "bn-en": "Helsinki-NLP/opus-mt-bn-en",
        # "en-od": "some-en-od-model",
        # "od-en": "some-od-en-model",
    }

    model_name = model_mappings.get(model_key)
    if not model_name:
        # Not implemented for bn/od or other languages - return helpful error
        raise ValueError(f"Translation model not available for {source_lang} -> {target_lang}")

    try:
        logger.info(f"Loading translation model: {model_name}")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        translation_models[model_key] = (model, tokenizer)
        logger.info(f"Translation model loaded: {model_name}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load translation model {model_name}: {str(e)}")
        raise

def transcribe_audio(filepath, language=None):
    """
    Transcribe audio file using Whisper (returns recognized text, romanized/dependent on model)
    """
    try:
        logger.info(f"Transcribing audio: {filepath}")
        if language:
            result = whisper_model.transcribe(filepath, language=language, task="transcribe")
        else:
            result = whisper_model.transcribe(filepath, task="transcribe")
        recognized_text = result.get("text", "").strip()
        detected_lang = result.get("language", "unknown")
        logger.info(f"Transcription complete. Detected language: {detected_lang}")
        return recognized_text
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise

def translate_text(text, source_lang, target_lang):
    """
    Translate text using HuggingFace MarianMT
    """
    try:
        logger.info(f"Translating text: {source_lang} -> {target_lang}")
        model, tokenizer = get_translation_model(source_lang, target_lang)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        logger.info("Translation complete")
        return translated_text
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise

def generate_speech(text, lang, output_path):
    """
    Generate speech using gTTS. If language unsupported, fallback to English.
    """
    try:
        logger.info(f"Generating speech for language: {lang}")
        gtts_lang_map = {
            "hi": "hi",   # Hindi
            "en": "en",   # English
            "bn": "bn",   # Bengali (gTTS supports 'bn' in many setups)
            # Odia not supported by gTTS - fallback to Hindi/English as appropriate
            "od": None,
        }
        gtts_lang = gtts_lang_map.get(lang, "en")
        if gtts_lang is None:
            logger.warning(f"gTTS does not support '{lang}'. Falling back to English for TTS.")
            gtts_lang = "en"
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        tts.save(output_path)
        logger.info(f"Speech generated: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"TTS generation error: {str(e)}")
        raise

def convert_to_wav_mono_16k(src_path):
    """
    Convert any audio file to 16kHz mono WAV using ffmpeg, if ffmpeg is available.
    Returns path to converted wav file (or original path on failure).
    """
    base = os.path.splitext(os.path.basename(src_path))[0]
    out_path = os.path.join(UPLOAD_FOLDER, f"{base}_16k_mono.wav")
    cmd = f'ffmpeg -y -i "{src_path}" -ac 1 -ar 16000 "{out_path}"'
    try:
        subprocess.run(shlex.split(cmd), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Converted audio to 16k mono WAV: {out_path}")
        return out_path
    except Exception as e:
        logger.warning(f"FFmpeg conversion failed or not available, using original file. Error: {e}")
        return src_path

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "whisper_loaded": whisper_model is not None,
        "translation_models_cached": list(translation_models.keys()),
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route("/translate", methods=["POST"])
def translate():
    """
    Main translation endpoint
    Expected form-data:
      - file: audio file (mp3/wav/ogg/m4a/flac)
      - input_lang: source language code (e.g., 'en', 'hi', 'bn', 'od')
      - target_lang: target language code (e.g., 'hi', 'en', 'bn', 'od')
    """
    try:
        # Validate file upload
        if "file" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": f"File type not allowed. Allowed: {ALLOWED_EXTENSIONS}"}), 400

        # Get language parameters
        input_lang = request.form.get("input_lang")
        target_lang = request.form.get("target_lang")
        if not input_lang or not target_lang:
            return jsonify({"error": "input_lang and target_lang are required"}), 400

        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = secure_filename(file.filename)
        filename = f"{timestamp}_{original_filename}"
        upload_filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_filepath)
        logger.info(f"File uploaded: {filename}")

        # Preprocess audio (16k mono wav) to improve ASR
        processed_audio = convert_to_wav_mono_16k(upload_filepath)

        recognized_text = ""
        translated_text = ""

        # Use Whisper direct translate for hi -> en (end-to-end), which often performs better
        if input_lang == "hi" and target_lang == "en":
            try:
                logger.info("Using Whisper end-to-end translate for hi->en")
                # End-to-end translation (Hindi audio -> English text)
                whisper_result = whisper_model.transcribe(processed_audio, language="hi", task="translate")
                translated_text = whisper_result.get("text", "").strip()
                # For recognized_text, also get Hindi transcription (romanized) if you want
                try:
                    raw_trans = whisper_model.transcribe(processed_audio, language="hi", task="transcribe")
                    recognized_text = raw_trans.get("text", "").strip()
                except Exception:
                    recognized_text = ""
            except Exception as e:
                logger.error(f"Whisper translate failed: {e}")
                return jsonify({"error": f"Whisper translate failed: {str(e)}"}), 500
        else:
            # Default flow: transcribe first, then translate (e.g., en->hi)
            try:
                recognized_text = transcribe_audio(processed_audio, language=input_lang)
            except Exception as e:
                return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

            if not recognized_text:
                return jsonify({"error": "No speech detected in audio"}), 400

            # For bn/od translation we currently return a helpful error (not implemented)
            try:
                translated_text = translate_text(recognized_text, input_lang, target_lang)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                return jsonify({"error": f"Translation failed: {str(e)}"}), 500

        # Generate TTS of translated text
        output_filename = f"{timestamp}_output.mp3"
        output_filepath = os.path.join(GENERATED_AUDIO_FOLDER, output_filename)
        try:
            generate_speech(translated_text, target_lang, output_filepath)
        except Exception as e:
            return jsonify({"error": f"TTS generation failed: {str(e)}"}), 500

        # Optional cleanup - keep or remove uploaded originals as you prefer
        try:
            if os.path.exists(upload_filepath):
                os.remove(upload_filepath)
                logger.info(f"Cleaned up upload: {upload_filepath}")
        except Exception as e:
            logger.warning(f"Failed to clean up upload: {str(e)}")

        audio_url = f"/audio/{output_filename}"
        response = {
            "recognized_text": recognized_text,
            "translated_text": translated_text,
            "audio_url": audio_url,
            "source_language": input_lang,
            "target_language": target_lang,
            "timestamp": timestamp
        }
        logger.info("Translation request completed successfully")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route("/audio/<filename>", methods=["GET"])
def serve_audio(filename):
    try:
        filename = secure_filename(filename)
        filepath = os.path.join(GENERATED_AUDIO_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "Audio file not found"}), 404
        return send_file(filepath, mimetype="audio/mpeg", as_attachment=False, download_name=filename)
    except Exception as e:
        logger.error(f"Error serving audio: {str(e)}")
        return jsonify({"error": "Failed to serve audio file"}), 500

@app.route("/languages", methods=["GET"])
def get_supported_languages():
    """
    Return list of supported language pairs (kept minimal: en, hi, bn, od)
    Note: translation models implemented currently only for en<->hi.
    """
    supported_pairs = [
        {"source": "en", "target": "hi", "name": "English to Hindi"},
        {"source": "hi", "target": "en", "name": "Hindi to English"},
        {"source": "en", "target": "bn", "name": "English to Bengali (not implemented)"},
        {"source": "bn", "target": "en", "name": "Bengali to English (not implemented)"},
        {"source": "en", "target": "od", "name": "English to Odia (not implemented)"},
        {"source": "od", "target": "en", "name": "Odia to English (not implemented)"},
    ]
    return jsonify({"supported_language_pairs": supported_pairs}), 200

if __name__ == "__main__":
    # Load Whisper model at startup
    load_whisper_model()
    # NOTE: translation models are loaded lazily when first requested to save startup time & disk.
    # If you prefer to preload en<->hi at startup, uncomment below:
    # try:
    #     get_translation_model("en", "hi")
    #     get_translation_model("hi", "en")
    # except Exception as e:
    #     logger.warning(f"Preloading translation models failed or skipped: {e}")

    # Run Flask app
    app.run(host="0.0.0.0", port=8000, debug=True)
