"""
app/processors/whisper_processor.py
===================================
Speech-to-text via Whisper (local or API).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from app.utils.logging import get_logger

logger = get_logger("processors.whisper")


class WhisperProcessor:
    def __init__(self, model_size: str = "base", device: Optional[str] = None,
                 language: Optional[str] = None, use_api: bool = False,
                 api_key: Optional[str] = None) -> None:
        self.model_size = model_size
        self.device = device or "cpu"
        self.language = language
        self.use_api = use_api
        self.api_key = api_key
        self._model = None

    def transcribe(self, audio_path: str) -> str:
        return self._transcribe_api(audio_path) if self.use_api else self._transcribe_local(audio_path)

    def _transcribe_local(self, audio_path: str) -> str:
        try:
            import whisper
            if self._model is None:
                self._model = whisper.load_model(self.model_size, device=self.device)
            return self._model.transcribe(audio_path, language=self.language).get("text", "").strip()
        except ImportError:
            logger.error("pip install openai-whisper")
            return ""
        except Exception as exc:
            logger.error("Whisper failed %s: %s", audio_path, exc)
            return ""

    def _transcribe_api(self, audio_path: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            with open(audio_path, "rb") as f:
                return client.audio.transcriptions.create(model="whisper-1", file=f, language=self.language).text.strip()
        except Exception as exc:
            logger.error("Whisper API failed %s: %s", audio_path, exc)
            return ""

    def extract_audio_from_video(self, video_path: str) -> Optional[str]:
        try:
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(video_path)
            if clip.audio is None:
                clip.close()
                return None
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            clip.audio.write_audiofile(tmp.name, logger=None)
            clip.close()
            return tmp.name
        except Exception as exc:
            logger.error("Audio extraction failed %s: %s", video_path, exc)
            return None
