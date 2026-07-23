"""
SpeechSync unified backend.

A single Flask server that runs speech-to-speech translation using
selectable, modular components:

    ASR  ->  Translation  ->  TTS

Three ready-made pipelines are exposed, plus a "custom" mode where the
Android app can mix and match any supported component.

Components
----------
ASR:
    whisper-medium      openai/whisper-medium
    whisper-large-v3    openai/whisper-large-v3
Translation:
    indictrans2         ai4bharat/indictrans2-*-1B  (best for Indic languages)
    nllb                facebook/nllb-200-3.3B      (broad multilingual)
TTS:
    gtts                Google TTS (online, light)
    mms                 facebook/mms-tts-*          (neural, native Indic voices)

Presets (see PIPELINES)
-----------------------
    balanced      whisper-large-v3 -> indictrans2 -> gtts
    multilingual  whisper-large-v3 -> nllb        -> gtts
    neural        whisper-large-v3 -> indictrans2 -> mms

Everything is lazy-loaded: a model is only pulled into GPU memory the
first time it is actually requested, then cached for reuse. This keeps
startup fast and memory usage proportional to what you actually use.

Target hardware: a single ~48GB GPU (all "best" models fit comfortably).
"""

import os
import threading
import subprocess
import shlex
import traceback
import logging
from datetime import datetime

import torch
import soundfile as sf
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("speechsync")

HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")

# ----------------------------------------------------------------------
# Flask app / paths
# ----------------------------------------------------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
GENERATED_AUDIO_FOLDER = os.path.join(BASE_DIR, "generated_audio")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_AUDIO_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp3", "wav", "ogg", "m4a", "flac"}

# ----------------------------------------------------------------------
# Device
# ----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if device == "cuda" else torch.float32
logger.info(f"Using device: {device}")

# ----------------------------------------------------------------------
# Language maps
#   App language codes -> per-model language codes.
#   Supported app codes: en, hi, bn, mr, od, ta, te, gu, kn, ml
# ----------------------------------------------------------------------
INDICTRANS_LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
    "od": "ory_Orya",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
}

NLLB_LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
    "od": "ory_Orya",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
}

GTTS_LANG_MAP = {
    "en": "en",
    "hi": "hi",
    "bn": "bn",
    "mr": "mr",
    "ta": "ta",
    "te": "te",
    "gu": "gu",
    "kn": "kn",
    "ml": "ml",
    # "od" (Odia) is not supported by gTTS -> handled via romanization fallback
}

MMS_LANG_MAP = {
    "en": "eng",
    "hi": "hin",
    "bn": "ben",
    "mr": "mar",
    "od": "ory",
    "ta": "tam",
    "te": "tel",
    "gu": "guj",
    "kn": "kan",
    "ml": "mal",
}

SUPPORTED_LANGS = set(INDICTRANS_LANG_MAP.keys())

# ----------------------------------------------------------------------
# Component + pipeline catalog (also served to the app via /pipelines)
# ----------------------------------------------------------------------
ASR_CHOICES = {
    "whisper-medium": "Whisper Medium",
    "whisper-large-v3": "Whisper Large v3",
}
TRANSLATION_CHOICES = {
    "indictrans2": "IndicTrans2 (1B)",
    "nllb": "NLLB-200 (3.3B)",
}
TTS_CHOICES = {
    "gtts": "gTTS",
    "mms": "MMS-TTS (Meta)",
}

PIPELINES = {
    "balanced": {
        "name": "Balanced Precision",
        "asr": "whisper-large-v3",
        "translation": "indictrans2",
        "tts": "gtts",
    },
    "multilingual": {
        "name": "Broad Multilingual",
        "asr": "whisper-large-v3",
        "translation": "nllb",
        "tts": "gtts",
    },
    "neural": {
        "name": "Natural Neural Voice",
        "asr": "whisper-large-v3",
        "translation": "indictrans2",
        "tts": "mms",
    },
}

DEFAULT_PIPELINE = "neural"

# ----------------------------------------------------------------------
# Model caches + locks (lazy loading)
# ----------------------------------------------------------------------
_lock = threading.Lock()
_asr_models = {}            # key: asr choice -> hf pipeline
_indictrans_models = {}     # key: "src-tgt direction" -> (model, tok, processor)
_nllb = {}                  # key: "model" -> (model, tokenizer)
_mms_models = {}            # key: mms lang -> (model, tokenizer)

ASR_MODEL_IDS = {
    "whisper-medium": "openai/whisper-medium",
    "whisper-large-v3": "openai/whisper-large-v3",
}


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_to_wav_mono_16k(src):
    """Normalise any input audio to mono 16k WAV for the ASR model."""
    base = os.path.splitext(os.path.basename(src))[0]
    out = os.path.join(UPLOAD_FOLDER, f"{base}_16k.wav")
    cmd = f'ffmpeg -y -i "{src}" -ac 1 -ar 16000 -acodec pcm_s16le "{out}"'
    try:
        subprocess.run(
            shlex.split(cmd),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
        return out
    except Exception:
        logger.warning("FFmpeg failed, using original audio")
        return src


# ----------------------------------------------------------------------
# ASR: Whisper
# ----------------------------------------------------------------------
def get_asr(choice):
    if choice not in ASR_MODEL_IDS:
        raise ValueError(f"Unsupported ASR component: {choice}")

    if choice in _asr_models:
        return _asr_models[choice]

    with _lock:
        if choice in _asr_models:
            return _asr_models[choice]

        from transformers import pipeline as hf_pipeline

        model_id = ASR_MODEL_IDS[choice]
        logger.info(f"Loading ASR '{choice}' ({model_id}) ...")
        asr = hf_pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            device=0 if device == "cuda" else -1,
            torch_dtype=TORCH_DTYPE,
            chunk_length_s=30,
            model_kwargs={"low_cpu_mem_usage": True},
        )
        asr.model.eval()
        _asr_models[choice] = asr
        logger.info(f"ASR '{choice}' ready")
        return asr


def transcribe(audio_path, asr_choice):
    asr = get_asr(asr_choice)
    with _lock:
        result = asr(audio_path, return_timestamps=False)
    return result["text"].strip()


# ----------------------------------------------------------------------
# Translation: IndicTrans2
# ----------------------------------------------------------------------
def _indictrans_model_name(src, tgt):
    if src == "en":
        return "ai4bharat/indictrans2-en-indic-1B"
    if tgt == "en":
        return "ai4bharat/indictrans2-indic-en-1B"
    return "ai4bharat/indictrans2-indic-indic-1B"


def get_indictrans(src, tgt):
    model_name = _indictrans_model_name(src, tgt)
    if model_name in _indictrans_models:
        return _indictrans_models[model_name]

    with _lock:
        if model_name in _indictrans_models:
            return _indictrans_models[model_name]

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from IndicTransToolkit import IndicProcessor

        logger.info(f"Loading IndicTrans2: {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=HF_TOKEN, trust_remote_code=True
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype=TORCH_DTYPE,
        )
        if device == "cuda":
            model = model.to(device)
        model.eval()
        processor = IndicProcessor(inference=True)

        _indictrans_models[model_name] = (model, tokenizer, processor)
        logger.info(f"IndicTrans2 '{model_name}' ready")
        return _indictrans_models[model_name]


def translate_indictrans(text, src, tgt):
    model, tokenizer, processor = get_indictrans(src, tgt)

    batch = processor.preprocess_batch(
        [text],
        src_lang=INDICTRANS_LANG_MAP[src],
        tgt_lang=INDICTRANS_LANG_MAP[tgt],
    )
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256, num_beams=5)

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return processor.postprocess_batch(decoded, lang=INDICTRANS_LANG_MAP[tgt])[0]


# ----------------------------------------------------------------------
# Translation: NLLB-200
# ----------------------------------------------------------------------
def get_nllb():
    model_name = "facebook/nllb-200-3.3B"
    if model_name in _nllb:
        return _nllb[model_name]

    with _lock:
        if model_name in _nllb:
            return _nllb[model_name]

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        logger.info(f"Loading NLLB: {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, token=HF_TOKEN, torch_dtype=TORCH_DTYPE
        )
        if device == "cuda":
            model = model.to(device)
        model.eval()

        _nllb[model_name] = (model, tokenizer)
        logger.info("NLLB ready")
        return _nllb[model_name]


def translate_nllb(text, src, tgt):
    model, tokenizer = get_nllb()

    tokenizer.src_lang = NLLB_LANG_MAP[src]
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(NLLB_LANG_MAP[tgt])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=256,
            num_beams=5,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


def translate(text, src, tgt, translation_choice):
    if src == tgt:
        return text
    if translation_choice == "indictrans2":
        return translate_indictrans(text, src, tgt)
    if translation_choice == "nllb":
        return translate_nllb(text, src, tgt)
    raise ValueError(f"Unsupported translation component: {translation_choice}")


# ----------------------------------------------------------------------
# TTS: gTTS
# ----------------------------------------------------------------------
def _odia_to_roman(text):
    from indic_transliteration.sanscript import transliterate

    roman = transliterate(text, "oriya", "itrans")
    return roman.replace("M", "n").replace("A", "a").replace(".", "")


def tts_gtts(text, lang, out_base):
    from gtts import gTTS

    out_path = out_base + ".mp3"
    if lang in GTTS_LANG_MAP:
        gTTS(text=text, lang=GTTS_LANG_MAP[lang]).save(out_path)
        return out_path

    if lang == "od":
        # gTTS has no Odia voice -> romanize and speak with the English voice.
        gTTS(text=_odia_to_roman(text), lang="en").save(out_path)
        return out_path

    raise ValueError(f"gTTS does not support language: {lang}")


# ----------------------------------------------------------------------
# TTS: MMS-TTS (Meta, native neural Indic voices)
# ----------------------------------------------------------------------
def get_mms(lang):
    if lang not in MMS_LANG_MAP:
        raise ValueError(f"MMS-TTS does not support language: {lang}")

    mms_lang = MMS_LANG_MAP[lang]
    if mms_lang in _mms_models:
        return _mms_models[mms_lang]

    with _lock:
        if mms_lang in _mms_models:
            return _mms_models[mms_lang]

        from transformers import VitsModel, AutoTokenizer

        model_id = f"facebook/mms-tts-{mms_lang}"
        logger.info(f"Loading MMS-TTS: {model_id} ...")
        model = VitsModel.from_pretrained(model_id, token=HF_TOKEN)
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
        if device == "cuda":
            model = model.to(device)
        model.eval()

        _mms_models[mms_lang] = (model, tokenizer)
        logger.info(f"MMS-TTS '{model_id}' ready")
        return _mms_models[mms_lang]


def tts_mms(text, lang, out_base):
    model, tokenizer = get_mms(lang)

    inputs = tokenizer(text, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        waveform = model(**inputs).waveform

    audio = waveform.squeeze().detach().cpu().float().numpy()
    out_path = out_base + ".wav"
    sf.write(out_path, audio, model.config.sampling_rate)
    return out_path


def synthesize(text, lang, tts_choice, out_base):
    if tts_choice == "gtts":
        return tts_gtts(text, lang, out_base)
    if tts_choice == "mms":
        return tts_mms(text, lang, out_base)
    raise ValueError(f"Unsupported TTS component: {tts_choice}")


# ----------------------------------------------------------------------
# Pipeline resolution
# ----------------------------------------------------------------------
def resolve_pipeline(form):
    """Work out which components to run from the request form.

    Priority:
      1. explicit asr / translation / tts fields (custom pipeline)
      2. a named `pipeline` preset
      3. the default preset
    """
    asr = form.get("asr")
    translation = form.get("translation")
    tts = form.get("tts")

    if asr and translation and tts:
        return asr, translation, tts

    preset_id = form.get("pipeline", DEFAULT_PIPELINE)
    preset = PIPELINES.get(preset_id, PIPELINES[DEFAULT_PIPELINE])
    return (
        asr or preset["asr"],
        translation or preset["translation"],
        tts or preset["tts"],
    )


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": device})


@app.route("/pipelines", methods=["GET"])
def pipelines():
    return jsonify(
        {
            "pipelines": PIPELINES,
            "components": {
                "asr": ASR_CHOICES,
                "translation": TRANSLATION_CHOICES,
                "tts": TTS_CHOICES,
            },
            "languages": sorted(SUPPORTED_LANGS),
            "default": DEFAULT_PIPELINE,
        }
    )


@app.route("/translate", methods=["POST"])
def translate_api():
    try:
        start = datetime.now()

        if "file" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        file = request.files["file"]
        src = request.form.get("input_lang")
        tgt = request.form.get("target_lang")

        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Invalid or unsupported audio file"}), 400
        if src not in SUPPORTED_LANGS or tgt not in SUPPORTED_LANGS:
            return jsonify({"error": "Unsupported language"}), 400

        asr_choice, translation_choice, tts_choice = resolve_pipeline(request.form)

        # Validate the requested components up front.
        if asr_choice not in ASR_CHOICES:
            return jsonify({"error": f"Unsupported ASR: {asr_choice}"}), 400
        if translation_choice not in TRANSLATION_CHOICES:
            return jsonify({"error": f"Unsupported translation: {translation_choice}"}), 400
        if tts_choice not in TTS_CHOICES:
            return jsonify({"error": f"Unsupported TTS: {tts_choice}"}), 400

        logger.info(
            f"Pipeline -> ASR={asr_choice}, MT={translation_choice}, "
            f"TTS={tts_choice} ({src} -> {tgt})"
        )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        upload = os.path.join(
            UPLOAD_FOLDER, f"{stamp}_{secure_filename(file.filename)}"
        )
        file.save(upload)

        audio = convert_to_wav_mono_16k(upload)
        recognized = transcribe(audio, asr_choice)
        translated = translate(recognized, src, tgt, translation_choice)

        out_base = os.path.join(GENERATED_AUDIO_FOLDER, f"{stamp}_out")
        out_audio = synthesize(translated, tgt, tts_choice, out_base)

        return jsonify(
            {
                "recognized_text": recognized,
                "translated_text": translated,
                "audio_url": f"/audio/{os.path.basename(out_audio)}",
                "pipeline": {
                    "asr": asr_choice,
                    "translation": translation_choice,
                    "tts": tts_choice,
                },
                "processing_time": (datetime.now() - start).total_seconds(),
            }
        )

    except Exception:
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal error"}), 500


@app.route("/audio/<name>")
def serve_audio(name):
    path = os.path.join(GENERATED_AUDIO_FOLDER, secure_filename(name))
    if not os.path.exists(path):
        return jsonify({"error": "Audio not found"}), 404
    mimetype = "audio/wav" if path.endswith(".wav") else "audio/mpeg"
    return send_file(path, mimetype=mimetype)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Models are lazy-loaded on first use, so startup is instant.
    app.run(host="0.0.0.0", port=8000, debug=False)
