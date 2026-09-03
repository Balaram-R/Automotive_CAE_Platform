"""
app/loaders/audio_loader.py
===========================
Audio → Whisper transcription.
"""

from __future__ import annotations

from app.loaders.base import BaseLoader
from app.loaders.loader_factory import LoaderFactory
from app.models.schemas import FileMetadata, RawDocument
from app.utils.logging import get_logger

logger = get_logger("loaders.audio")


class AudioLoader(BaseLoader):
    supported_extensions = [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma"]

    def load(self, filepath: str, metadata: FileMetadata, **kw) -> list[RawDocument]:
        whisper_model = kw.get("whisper_model_size", "base")
        whisper_device = kw.get("whisper_device")
        whisper_lang  = kw.get("whisper_language")

        try:
            import whisper
            model = whisper.load_model(whisper_model, device=whisper_device or "cpu")
            result = model.transcribe(filepath, language=whisper_lang)
            text = result.get("text", "").strip()
            if text:
                return [RawDocument(content=text, metadata=metadata, source_loader="AudioLoader/Whisper")]
        except ImportError:
            logger.error("pip install openai-whisper")
        except Exception as exc:
            logger.error("Whisper failed %s: %s", filepath, exc)
        return []


LoaderFactory.register_many(AudioLoader.supported_extensions, AudioLoader())
