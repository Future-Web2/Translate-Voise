from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


def _number(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} in environment variables")
    return value


@dataclass(frozen=True)
class Settings:
    telegram_token: str = field(repr=False)
    translation_key: str = field(repr=False)
    qa_key: str = field(repr=False)
    allowed_users: frozenset[int]
    translation_model: str
    qa_model: str
    io_base_url: str = "https://api.intelligence.io.solutions/api/v1"
    whisper_model: str = "large-v3"
    cpu_threads: int = 2
    cache_dir: Path = Path("data/models")
    language: str | None = None
    glossary: str = ""
    asr_timeout: int = 900
    max_duration: int = 300
    max_file_bytes: int = 20 * 1024 * 1024
    queue_size: int = 5
    cooldown: int = 5
    port: int = 8080

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(override=False)
        ids = os.getenv("ALLOWED_USER_IDS", "")
        try:
            users = frozenset(int(item.strip()) for item in ids.split(",") if item.strip())
        except ValueError as exc:
            raise ValueError("ALLOWED_USER_IDS must contain numeric user IDs") from exc
        if any(uid <= 0 for uid in users):
            raise ValueError("ALLOWED_USER_IDS must contain positive user IDs")
        base = os.getenv("IO_BASE_URL", cls.io_base_url).rstrip("/")
        if urlparse(base).scheme != "https" or not urlparse(base).hostname:
            raise ValueError("IO_BASE_URL must use HTTPS")
        model = os.getenv("WHISPER_MODEL", "large-v3").strip()
        if model.endswith(".en") or "distil" in model.lower():
            raise ValueError("Use a multilingual Whisper model for Uzbek")
        language = os.getenv("ASR_LANGUAGE", "auto").strip().lower()
        if language not in {"auto", "uz", "ru", "en"}:
            raise ValueError("ASR_LANGUAGE must be auto, uz, ru or en")
        return cls(
            telegram_token=_required("TELEGRAM_BOT_TOKEN"),
            translation_key=_required("IO_TRANSLATION_API_KEY"),
            qa_key=_required("IO_QA_API_KEY"),
            allowed_users=users,
            translation_model=_required("IO_TRANSLATION_MODEL"),
            qa_model=_required("IO_QA_MODEL"),
            io_base_url=base,
            whisper_model=model,
            cpu_threads=_number("WHISPER_CPU_THREADS", 2, 1, 16),
            cache_dir=Path(os.getenv("MODEL_CACHE_DIR", "data/models")),
            language=None if language == "auto" else language,
            glossary=os.getenv("ASR_GLOSSARY", "")[:800],
            asr_timeout=_number("ASR_TIMEOUT_SECONDS", 900, 30, 3600),
            max_duration=_number("MAX_DURATION_SECONDS", 300, 1, 1200),
            max_file_bytes=_number("MAX_FILE_MB", 20, 1, 20) * 1024 * 1024,
            queue_size=_number("MAX_QUEUE_SIZE", 5, 1, 50),
            cooldown=_number("USER_COOLDOWN_SECONDS", 5, 0, 3600),
            port=_number("PORT", 8080, 1, 65535),
        )
