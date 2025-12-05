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
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac'}

# Global model variables (loaded once at startup)
whisper_model = None
translation_models = {}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_whisper_model():
    """Load Whisper model once at startup"""
    global whisper_model
    try:
        logger.info("Loading Whisper model (base)...")
        # Use 'base' for faster processing, can upgrade to 'medium' or 'large' later
        whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {str(e)}")
        raise

def get_translation_model(source_lang, target_lang):
    """
    Load translation model with fallback logic
    Returns (model, tokenizer) tuple
    """
    global translation_models
    
    # Create model key
    model_key = f"{source_lang}-{target_lang}"
    
    # Return cached model if already loaded
    if model_key in translation_models:
        return translation_models[model_key]
    
    # Define model name based on language pair
    # HuggingFace Helsinki-NLP models for common pairs
    model_mappings = {
        "en-hi": "Helsinki-NLP/opus-mt-en-hi",
        "hi-en": "Helsinki-NLP/opus-mt-hi-en",
        "en-es": "Helsinki-NLP/opus-mt-en-es",
        "es-en": "Helsinki-NLP/opus-mt-es-en",
        "en-fr": "Helsinki-NLP/opus-mt-en-fr",
        "fr-en": "Helsinki-NLP/opus-mt-fr-en",
        "en-de": "Helsinki-NLP/opus-mt-en-de",
        "de-en": "Helsinki-NLP/opus-mt-de-en",
    }
    
    model_name = model_mappings.get(model_key)
    
    if not model_name:
        raise ValueError(f"Translation model not available for {source_lang} -> {target_lang}")
    
    try:
        logger.info(f"Loading translation model: {model_name}")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        # Cache the model
        translation_models[model_key] = (model, tokenizer)
        logger.info(f"Translation model loaded: {model_name}")
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load translation model {model_name}: {str(e)}")
        raise

def transcribe_audio(filepath, language=None):
    """
    Transcribe audio file using Whisper
    Args:
        filepath: path to audio file
        language: language code (optional, Whisper can auto-detect)
    Returns:
        recognized text string
    """
    try:
        logger.info(f"Transcribing audio: {filepath}")
        
        # Whisper can auto-detect language if not specified
        if language:
            result = whisper_model.transcribe(filepath, language=language)
        else:
            result = whisper_model.transcribe(filepath)
        
        recognized_text = result["text"].strip()
        detected_lang = result.get("language", "unknown")
        
        logger.info(f"Transcription complete. Detected language: {detected_lang}")
        return recognized_text
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise

def translate_text(text, source_lang, target_lang):
    """
    Translate text using HuggingFace MarianMT
    Args:
        text: input text
        source_lang: source language code
        target_lang: target language code
    Returns:
        translated text string
    """
    try:
        logger.info(f"Translating: {source_lang} -> {target_lang}")
        
        model, tokenizer = get_translation_model(source_lang, target_lang)
        
        # Tokenize and translate
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
    Generate speech using gTTS
    Args:
        text: text to convert to speech
        lang: language code (gTTS format)
        output_path: path to save MP3 file
    Returns:
        output_path
    """
    try:
        logger.info(f"Generating speech for language: {lang}")
        
        # gTTS language code mapping (ISO 639-1)
        gtts_lang_map = {
            "hi": "hi",  # Hindi
            "en": "en",  # English
            "es": "es",  # Spanish
            "fr": "fr",  # French
            "de": "de",  # German
            "bn": "bn",  # Bengali
            "ta": "ta",  # Tamil
            "te": "te",  # Telugu
        }
        
        # Get gTTS language code or fallback to English
        gtts_lang = gtts_lang_map.get(lang, "en")
        
        if gtts_lang != lang:
            logger.warning(f"Language {lang} not supported by gTTS, falling back to English")
        
        # Generate TTS
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        tts.save(output_path)
        
        logger.info(f"Speech generated: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"TTS generation error: {str(e)}")
        raise


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "whisper_loaded": whisper_model is not None,
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route("/translate", methods=["POST"])
def translate():
    """
    Main translation endpoint
    Expected form-data:
      - file: audio file (mp3/wav/ogg/m4a/flac)
      - input_lang: source language code (e.g., 'en', 'hi')
      - target_lang: target language code (e.g., 'hi', 'en')
    
    Returns JSON:
      - recognized_text: transcribed text
      - translated_text: translated text
      - audio_url: URL to download generated audio
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
        
        # Save uploaded file with secure filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = secure_filename(file.filename)
        filename = f"{timestamp}_{original_filename}"
        upload_filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_filepath)
        
        logger.info(f"File uploaded: {filename}")
        
        # Step 1: Transcribe audio using Whisper
        try:
            recognized_text = transcribe_audio(upload_filepath, language=input_lang)
        except Exception as e:
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
        
        if not recognized_text:
            return jsonify({"error": "No speech detected in audio"}), 400
        
        # Step 2: Translate text
        try:
            translated_text = translate_text(recognized_text, input_lang, target_lang)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"Translation failed: {str(e)}"}), 500
        
        # Step 3: Generate TTS audio
        output_filename = f"{timestamp}_output.mp3"
        output_filepath = os.path.join(GENERATED_AUDIO_FOLDER, output_filename)
        
        try:
            generate_speech(translated_text, target_lang, output_filepath)
        except Exception as e:
            return jsonify({"error": f"TTS generation failed: {str(e)}"}), 500
        
        # Create audio URL (relative to server)
        audio_url = f"/audio/{output_filename}"
        
        # Clean up uploaded file (optional - comment out if you want to keep uploads)
        try:
            os.remove(upload_filepath)
            logger.info(f"Cleaned up upload: {upload_filepath}")
        except Exception as e:
            logger.warning(f"Failed to clean up upload: {str(e)}")
        
        # Return response
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
    """
    Serve generated audio files
    Args:
        filename: name of the audio file
    Returns:
        MP3 file for streaming/download
    """
    try:
        # Secure the filename to prevent directory traversal
        filename = secure_filename(filename)
        filepath = os.path.join(GENERATED_AUDIO_FOLDER, filename)
        
        if not os.path.exists(filepath):
            return jsonify({"error": "Audio file not found"}), 404
        
        return send_file(
            filepath,
            mimetype="audio/mpeg",
            as_attachment=False,  # Stream instead of forcing download
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error serving audio: {str(e)}")
        return jsonify({"error": "Failed to serve audio file"}), 500


@app.route("/languages", methods=["GET"])
def get_supported_languages():
    """
    Return list of supported language pairs
    """
    supported_pairs = [
        {"source": "en", "target": "hi", "name": "English to Hindi"},
        {"source": "hi", "target": "en", "name": "Hindi to English"},
        {"source": "en", "target": "es", "name": "English to Spanish"},
        {"source": "es", "target": "en", "name": "Spanish to English"},
        {"source": "en", "target": "fr", "name": "English to French"},
        {"source": "fr", "target": "en", "name": "French to English"},
        {"source": "en", "target": "de", "name": "English to German"},
        {"source": "de", "target": "en", "name": "German to English"},
    ]
    return jsonify({"supported_language_pairs": supported_pairs}), 200


if __name__ == "__main__":
    # Load Whisper model at startup
    load_whisper_model()
    
    # Run Flask app
    app.run(host="0.0.0.0", port=8000, debug=True)