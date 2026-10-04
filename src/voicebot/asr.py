"""Serialized ASR jobs in a reusable process that can be forcibly stopped."""

from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import Settings

MAX_RESPONSE_BYTES = 1024 * 1024
MAX_SEGMENTS = 2000
MAX_WORDS = 15000


class ASRError(Exception):
    """Safe public error; never contains media, model output, or secrets."""


class LocalASR:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._state = "cold"
        self._closed = False
        self._worker_settings = {
            "model": settings.whisper_model,
            "cpu_threads": settings.cpu_threads,
            "cache_dir": str(settings.cache_dir.resolve()),
            "language": settings.language,
            "glossary": settings.glossary,
        }

    @property
    def state(self) -> str:
        return self._state

    async def _stop_process(self) -> None:
        process, self._process = self._process, None
        self._state = "cold"
        if process is None:
            return
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        # Reap even on timeout/cancellation; do not leave a CPU-consuming child.
        await asyncio.shield(process.wait())

    async def _request(self, path: Path) -> list[dict[str, Any]]:
        if self._process is None or self._process.returncode is not None:
            environment = {
                key: value
                for key, value in os.environ.items()
                if key not in {"TELEGRAM_BOT_TOKEN", "IO_TRANSLATION_API_KEY", "IO_QA_API_KEY"}
            }
            self._process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "voicebot.asr_worker",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=environment,
                limit=MAX_RESPONSE_BYTES,
                **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}),
            )
        if self._closed:
            raise ASRError("Распознавание остановлено.")
        process = self._process
        assert process.stdin is not None and process.stdout is not None
        payload = {
            "path": str(await asyncio.to_thread(path.resolve)),
            "settings": self._worker_settings,
        }
        process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        await process.stdin.drain()
        raw = await process.stdout.readline()
        if not raw or len(raw) > MAX_RESPONSE_BYTES:
            raise ASRError("Не удалось распознать запись.")
        response = json.loads(raw)
        if not isinstance(response, dict) or response.get("ok") is not True:
            raise ASRError("Не удалось распознать запись.")
        return _validate_segments(response.get("segments"))

    async def transcribe(self, path: Path) -> list[dict[str, Any]]:
        # Cancellation while waiting for the lock must not abort another job.
        async with self._lock:
            if self._closed:
                raise ASRError("Распознавание остановлено.")
            self._state = "working"
            try:
                segments = await asyncio.wait_for(self._request(path), self.settings.asr_timeout)
                self._state = "ready"
                return segments
            except asyncio.CancelledError:
                await self._stop_process()
                raise
            except TimeoutError:
                await self._stop_process()
                raise ASRError(
                    "Распознавание заняло слишком много времени. Попробуйте запись короче."
                ) from None
            except Exception:
                await self._stop_process()
                raise ASRError("Не удалось распознать запись.") from None

    async def close(self) -> None:
        self._closed = True
        await self._stop_process()


def _validate_segments(value: Any) -> list[dict[str, Any]]:
    """Reject corrupted or unbounded worker output at the process boundary."""
    if not isinstance(value, list) or len(value) > MAX_SEGMENTS:
        raise ValueError("Invalid ASR response")
    word_count = 0
    for segment in value:
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise ValueError("Invalid ASR segment")
        for key in ("start", "end", "avg_logprob"):
            if not _finite_number(segment.get(key)):
                raise ValueError("Invalid ASR metadata")
        if segment["start"] < 0 or segment["end"] < segment["start"]:
            raise ValueError("Invalid ASR timestamps")
        words = segment.get("words")
        if not isinstance(words, list):
            raise ValueError("Invalid ASR words")
        word_count += len(words)
        if word_count > MAX_WORDS:
            raise ValueError("Too many ASR words")
        for word in words:
            if not isinstance(word, dict) or not isinstance(word.get("word"), str):
                raise ValueError("Invalid ASR word")
            for key in ("start", "end", "probability"):
                if not _finite_number(word.get(key)):
                    raise ValueError("Invalid ASR word metadata")
            if (
                not 0 <= word["probability"] <= 1
                or word["start"] < 0
                or word["end"] < word["start"]
            ):
                raise ValueError("Invalid ASR word metadata")
    return value


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
