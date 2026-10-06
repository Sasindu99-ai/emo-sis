"""Speech transcription module utilizing OpenAI Whisper with Google Speech Recognition fallback."""

import os
import sys
from typing import Optional

try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

try:
    import speech_recognition as sr
    HAS_SPEECH_RECOGNITION = True
except ImportError:
    HAS_SPEECH_RECOGNITION = False

_whisper_model = None


def get_whisper_model(model_name: str = "tiny"):
    """Loads and caches Whisper model (default 'tiny' for low latency)."""
    global _whisper_model
    if _whisper_model is None and HAS_WHISPER:
        try:
            _whisper_model = whisper.load_model(model_name)
        except Exception as e:
            print(f"[Warning] Failed to load Whisper ({model_name}): {e}", file=sys.stderr)
    return _whisper_model


def transcribe_whisper(audio_path: str, model_name: str = "tiny") -> Optional[str]:
    """Transcribes audio using local Whisper model."""
    if not os.path.exists(audio_path):
        return None
    model = get_whisper_model(model_name)
    if model is None:
        return None
    try:
        result = model.transcribe(audio_path, fp16=False)
        return result.get("text", "").strip()
    except Exception as e:
        print(f"[Whisper error]: {e}", file=sys.stderr)
        return None


def transcribe_google(audio_path: str) -> Optional[str]:
    """Fallback transcription using Google Speech Recognition API."""
    if not HAS_SPEECH_RECOGNITION or not os.path.exists(audio_path):
        return None
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data).strip()
    except Exception:
        return None


def transcribe(audio_path: str, prefer_whisper: bool = True) -> str:
    """Transcribes audio file to text using Whisper with Google fallback.
    
    Returns:
        Transcribed string, or empty string if no speech detected.
    """
    if prefer_whisper:
        text = transcribe_whisper(audio_path)
        if text:
            return text
        text = transcribe_google(audio_path)
        if text:
            return text
    else:
        text = transcribe_google(audio_path)
        if text:
            return text
        text = transcribe_whisper(audio_path)
        if text:
            return text
    return ""
