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
import threading

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

# Set device for faster processing
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_whisper_model():
    """Load Whisper model once at startup"""
    global whisper_model
    try:
        logger.info("Loading Whisper model (base)...")
        # Use base model for faster processing (change to 'small' for better accuracy)
        whisper_model = whisper.load_model("base", device=device)
        logger.info(f"Whisper model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {str(e)}")
        raise

def get_translation_model(source_lang, target_lang):
    """
    Load translation model with fallback logic.
    Returns (model, tokenizer) tuple.
    """
    global translation_models
    model_key = f"{source_lang}-{target_lang}"
    if model_key in translation_models:
        return translation_models[model_key]

    model_mappings = {
        "en-hi": "Helsinki-NLP/opus-mt-en-hi",
        "hi-en": "Helsinki-NLP/opus-mt-hi-en",
    }

    model_name = model_mappings.get(model_key)
    if not model_name:
        raise ValueError(f"Translation model not available for {source_lang} -> {target_lang}")

    try:
        logger.info(f"Loading translation model: {model_name}")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        # Move model to GPU if available
        if device == "cuda":
            model = model.to(device)
        
        translation_models[model_key] = (model, tokenizer)
        logger.info(f"Translation model loaded: {model_name} on {device}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load translation model {model_name}: {str(e)}")
        raise

def transcribe_audio(filepath, language=None):
    """
    Transcribe audio file using Whisper
    """
    try:
        logger.info(f"Transcribing audio: {filepath}")
        # Use fp16 for faster inference on GPU
        fp16 = device == "cuda"
        
        if language:
            result = whisper_model.transcribe(
                filepath, 
                language=language, 
                task="transcribe",
                fp16=fp16
            )
        else:
            result = whisper_model.transcribe(
                filepath, 
                task="transcribe",
                fp16=fp16
            )
        
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
        
        # Prepare inputs
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # Move inputs to device
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():  # Disable gradient calculation for faster inference
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
    """
    try:
        logger.info(f"Generating speech for language: {lang}")
        gtts_lang_map = {
            "hi": "hi",
            "en": "en",
            "bn": "bn",
            "od": None,
        }
        gtts_lang = gtts_lang_map.get(lang, "en")
        if gtts_lang is None:
            logger.warning(f"gTTS does not support '{lang}'. Falling back to English.")
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
    Convert audio to 16kHz mono WAV using ffmpeg
    """
    base = os.path.splitext(os.path.basename(src_path))[0]
    out_path = os.path.join(UPLOAD_FOLDER, f"{base}_16k_mono.wav")
    
    # Faster conversion with reduced quality checks
    cmd = f'ffmpeg -y -i "{src_path}" -ac 1 -ar 16000 -acodec pcm_s16le "{out_path}"'
    try:
        subprocess.run(
            shlex.split(cmd), 
            check=True, 
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30  # Timeout after 30 seconds
        )
        logger.info(f"Converted audio to 16k mono WAV: {out_path}")
        return out_path
    except subprocess.TimeoutExpired:
        logger.warning(f"FFmpeg conversion timed out, using original file")
        return src_path
    except Exception as e:
        logger.warning(f"FFmpeg conversion failed, using original file. Error: {e}")
        return src_path

def cleanup_old_files():
    """
    Clean up old files (older than 1 hour) in background
    """
    def cleanup():
        import time
        current_time = time.time()
        for folder in [UPLOAD_FOLDER, GENERATED_AUDIO_FOLDER]:
            try:
                for filename in os.listdir(folder):
                    filepath = os.path.join(folder, filename)
                    if os.path.isfile(filepath):
                        file_age = current_time - os.path.getmtime(filepath)
                        if file_age > 3600:  # 1 hour
                            os.remove(filepath)
                            logger.info(f"Cleaned up old file: {filepath}")
            except Exception as e:
                logger.error(f"Cleanup error in {folder}: {e}")
    
    thread = threading.Thread(target=cleanup, daemon=True)
    thread.start()

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "device": device,
        "whisper_loaded": whisper_model is not None,
        "translation_models_cached": list(translation_models.keys()),
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route("/translate", methods=["POST"])
def translate():
    """
    Main translation endpoint
    """
    try:
        start_time = datetime.now()
        
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

        # Preprocess audio
        processed_audio = convert_to_wav_mono_16k(upload_filepath)

        recognized_text = ""
        translated_text = ""

        # Use Whisper direct translate for hi -> en
        if input_lang == "hi" and target_lang == "en":
            try:
                logger.info("Using Whisper end-to-end translate for hi->en")
                fp16 = device == "cuda"
                whisper_result = whisper_model.transcribe(
                    processed_audio, 
                    language="hi", 
                    task="translate",
                    fp16=fp16
                )
                translated_text = whisper_result.get("text", "").strip()
                
                # Get Hindi transcription
                try:
                    raw_trans = whisper_model.transcribe(
                        processed_audio, 
                        language="hi", 
                        task="transcribe",
                        fp16=fp16
                    )
                    recognized_text = raw_trans.get("text", "").strip()
                except Exception:
                    recognized_text = ""
            except Exception as e:
                logger.error(f"Whisper translate failed: {e}")
                return jsonify({"error": f"Whisper translate failed: {str(e)}"}), 500
        else:
            # Default flow: transcribe then translate
            try:
                recognized_text = transcribe_audio(processed_audio, language=input_lang)
            except Exception as e:
                return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

            if not recognized_text:
                return jsonify({"error": "No speech detected in audio"}), 400

            try:
                translated_text = translate_text(recognized_text, input_lang, target_lang)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                return jsonify({"error": f"Translation failed: {str(e)}"}), 500

        # Generate TTS
        output_filename = f"{timestamp}_output.mp3"
        output_filepath = os.path.join(GENERATED_AUDIO_FOLDER, output_filename)
        try:
            generate_speech(translated_text, target_lang, output_filepath)
        except Exception as e:
            return jsonify({"error": f"TTS generation failed: {str(e)}"}), 500

        # Cleanup uploaded file
        try:
            if os.path.exists(upload_filepath):
                os.remove(upload_filepath)
            if processed_audio != upload_filepath and os.path.exists(processed_audio):
                os.remove(processed_audio)
        except Exception as e:
            logger.warning(f"Failed to clean up files: {str(e)}")

        # Schedule background cleanup of old files
        cleanup_old_files()

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Total processing time: {processing_time:.2f} seconds")

        audio_url = f"/audio/{output_filename}"
        response = {
            "recognized_text": recognized_text,
            "translated_text": translated_text,
            "audio_url": audio_url,
            "source_language": input_lang,
            "target_language": target_lang,
            "timestamp": timestamp,
            "processing_time_seconds": round(processing_time, 2)
        }
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
        return send_file(
            filepath, 
            mimetype="audio/mpeg", 
            as_attachment=False, 
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error serving audio: {str(e)}")
        return jsonify({"error": "Failed to serve audio file"}), 500

@app.route("/languages", methods=["GET"])
def get_supported_languages():
    """Return list of supported language pairs"""
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
    
    # Preload translation models for better first-request performance
    try:
        logger.info("Preloading translation models...")
        get_translation_model("en", "hi")
        get_translation_model("hi", "en")
        logger.info("Translation models preloaded successfully")
    except Exception as e:
        logger.warning(f"Preloading translation models failed: {e}")

    # Run Flask app with production settings
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)