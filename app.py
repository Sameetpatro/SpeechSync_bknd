from flask import Flask, request, jsonify
import os
from datetime import datetime

app = Flask(__name__)

# folder to store uploaded audio temporarily
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/translate", methods=["POST"])
def translate():
    """
    Expected form-data:
      - file: audio file (mp3/wav)
      - input_lang: e.g. 'en', 'hi'
      - target_lang: e.g. 'hi', 'en'
    """
    if "file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["file"]
    input_lang = request.form.get("input_lang", None)
    target_lang = request.form.get("target_lang", None)

    if not input_lang or not target_lang:
        return jsonify({"error": "input_lang and target_lang are required"}), 400

    # Save file temporarily
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # TODO: Here later you will:
    #  1. Run ASR (speech -> text)
    #  2. Translate text
    #  3. Run TTS (text -> audio)
    # For now, we just fake it.

    dummy_recognized = f"Dummy recognized text in {input_lang.upper()} from audio {filename}"
    dummy_translated = f"Dummy translated text in {target_lang.upper()}"

    # In future: generate an mp3 file and give its URL
    dummy_audio_url = f"https://your-server.com/audio/{filename}.mp3"  # placeholder

    response = {
        "recognized_text": dummy_recognized,
        "translated_text": dummy_translated,
        "audio_url": dummy_audio_url
    }

    return jsonify(response), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
