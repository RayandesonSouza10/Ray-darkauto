from __future__ import annotations

from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("APP_DB_PATH", str(BASE_DIR / "app" / "data" / "ray_darkauto.db")))
OUTPUTS_DIR = Path(os.getenv("APP_OUTPUTS_DIR", str(BASE_DIR / "app" / "outputs")))
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1").strip()
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-flash").strip()  # backward compatibility
VERTEX_SCRIPT_MODEL = os.getenv("VERTEX_SCRIPT_MODEL", VERTEX_MODEL or "gemini-2.5-flash").strip()
VERTEX_IMAGE_MODEL = os.getenv("VERTEX_IMAGE_MODEL", "gemini-2.5-flash-image").strip()
VERTEX_VIDEO_MODEL = os.getenv("VERTEX_VIDEO_MODEL", "veo-3.0-generate-preview").strip()
VERTEX_API_VERSION = os.getenv("VERTEX_API_VERSION", "v1").strip()
TTS_VOICE_NAME = os.getenv("TTS_VOICE_NAME", "pt-BR-Standard-A").strip()
TTS_LANGUAGE_CODE = os.getenv("TTS_LANGUAGE_CODE", "pt-BR").strip()
TTS_AUDIO_FORMAT = os.getenv("TTS_AUDIO_FORMAT", "mp3").strip().lower()
TTS_MODEL_NAME = os.getenv("TTS_MODEL_NAME", "chirp-3.0-generate-001").strip()
TTS_SPEAKING_RATE = float(os.getenv("TTS_SPEAKING_RATE", "1.0"))
TTS_PITCH = float(os.getenv("TTS_PITCH", "0.0"))
TTS_VOLUME_GAIN_DB = float(os.getenv("TTS_VOLUME_GAIN_DB", "0.0"))
TTS_SAMPLE_RATE_HZ = int(os.getenv("TTS_SAMPLE_RATE_HZ", "44100"))
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg").strip()
VIDEO_WIDTH = int(os.getenv("VIDEO_WIDTH", "1280"))
VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "720"))
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "30"))

DB_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
