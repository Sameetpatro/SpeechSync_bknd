# SpeechSync Backend

A single Flask server that powers speech-to-speech translation for the
SpeechSync Android app. It runs a modular pipeline:

```
ASR  ->  Translation  ->  TTS
```

Components can be swapped independently, so the app can offer several
ready-made pipelines plus a fully custom one.

## Components

| Stage | Option | Model |
|-------|--------|-------|
| ASR | `whisper-medium` | `openai/whisper-medium` |
| ASR | `whisper-large-v3` | `openai/whisper-large-v3` |
| Translation | `indictrans2` | `ai4bharat/indictrans2-*-1B` (best for Indic) |
| Translation | `nllb` | `facebook/nllb-200-3.3B` (broad multilingual) |
| TTS | `gtts` | Google TTS (online) |
| TTS | `mms` | `facebook/mms-tts-*` (neural, native Indic voices) |

## Preset pipelines

| Preset | ASR | Translation | TTS |
|--------|-----|-------------|-----|
| `neural` (Natural Neural Voice) **default** | Whisper Large v3 | IndicTrans2 | MMS-TTS |
| `balanced` (Balanced Precision) | Whisper Large v3 | IndicTrans2 | gTTS |
| `multilingual` (Broad Multilingual) | Whisper Large v3 | NLLB-200 | gTTS |

Every model is **lazy-loaded**: it is only pulled into GPU memory the first
time it is requested, then cached. Startup is instant; memory grows only with
what you actually use.

## Supported languages

`en, hi, bn, mr, od, ta, te, gu, kn, ml`

## Setup

```bash
cd speechSync-backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# ffmpeg is required at the system level
#   Ubuntu/Debian: sudo apt-get install -y ffmpeg
#   macOS (brew):  brew install ffmpeg

# Some models (IndicTrans2 / NLLB) may need a Hugging Face token:
export HUGGINGFACE_HUB_TOKEN=hf_xxx      # optional

python app.py         # serves on 0.0.0.0:8000
```

## API

### `POST /translate`
Multipart form:

| Field | Required | Description |
|-------|----------|-------------|
| `file` | yes | Audio file (`mp3, wav, ogg, m4a, flac`) |
| `input_lang` | yes | Source language code |
| `target_lang` | yes | Target language code |
| `pipeline` | no | Preset id: `balanced` \| `multilingual` \| `neural` |
| `asr` | no | Override ASR component |
| `translation` | no | Override translation component |
| `tts` | no | Override TTS component |

If `asr` + `translation` + `tts` are all supplied, that custom combination is
used. Otherwise the named `pipeline` preset is used (default: `neural`).

Response:

```json
{
  "recognized_text": "...",
  "translated_text": "...",
  "audio_url": "/audio/20260101_120000_out.mp3",
  "pipeline": { "asr": "...", "translation": "...", "tts": "..." },
  "processing_time": 3.4
}
```

### `GET /pipelines`
Returns the full catalog of presets, components and supported languages.

### `GET /health`
Basic liveness check (`{"status": "ok", "device": "cuda"}`).

### `GET /audio/<name>`
Serves generated audio referenced by `audio_url`.

## Disk cleanup

The server does **not** keep uploads. After ASR finishes, the uploaded recording
and any ffmpeg temp WAV are deleted immediately.

Generated TTS audio is kept in a **FIFO queue of max 20 files** under
`generated_audio/`. When a 21st file is created, the oldest is deleted.

The Android app downloads that audio onto the device (`cache/translated_audio/`)
and also keeps at most **20 local files**, deleting the oldest when over the limit.
Playback uses the local file, not a live server stream.
