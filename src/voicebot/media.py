"""Bounded, cancellable conversion of Telegram media to speech-ready audio."""

from __future__ import annotations

import asyncio
import json
import math
import wave
from pathlib import Path


class MediaError(Exception):
    """The supplied media cannot be safely decoded as supported audio."""


_PROCESS_TIMEOUT_SECONDS = 60
_SAMPLE_RATE = 16_000
_BYTES_PER_SECOND = _SAMPLE_RATE * 2
_WAV_HEADER_ALLOWANCE = 4096
_DURATION_TOLERANCE_SECONDS = 0.25


async def _kill_and_reap(process: asyncio.subprocess.Process) -> None:
    """Do not leave a transcoder alive, including under repeated cancellation."""
    try:
        process.kill()
    except ProcessLookupError:
        pass
    waiter = asyncio.create_task(process.wait())
    while not waiter.done():
        try:
            await asyncio.shield(waiter)
        except asyncio.CancelledError:
            # The caller will re-raise its original cancellation after cleanup.
            continue
    waiter.result()


async def _run(*arguments: str) -> bytes:
    try:
        process = await asyncio.create_subprocess_exec(
            *arguments,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise MediaError("Audio tools could not be started.") from exc
    try:
        stdout, _stderr = await asyncio.wait_for(
            process.communicate(), timeout=_PROCESS_TIMEOUT_SECONDS
        )
    except TimeoutError as exc:
        await _kill_and_reap(process)
        raise MediaError("Audio processing exceeded the time limit.") from exc
    except asyncio.CancelledError:
        await _kill_and_reap(process)
        raise
    if process.returncode != 0:
        # Decoder diagnostics can contain user paths or arbitrary media metadata.
        raise MediaError("Audio could not be decoded.")
    return stdout


def _positive_duration(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        duration = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return duration if math.isfinite(duration) and duration > 0 else None


async def _probe_duration(source: Path) -> float:
    output = await _run(
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,duration",
        "-of",
        "json",
        str(source),
    )
    try:
        metadata = json.loads(output)
        streams = metadata.get("streams", [])
        audio_stream = next(stream for stream in streams if stream.get("codec_type") == "audio")
        duration = _positive_duration(audio_stream.get("duration"))
        if duration is None:
            duration = _positive_duration(metadata.get("format", {}).get("duration"))
        if duration is None:
            raise ValueError("Missing audio duration")
        return duration
    except (ValueError, TypeError, AttributeError, StopIteration) as exc:
        raise MediaError("Media must contain audio with a known positive duration.") from exc


def _validate_wav(destination: Path, max_bytes: int) -> float:
    try:
        if destination.stat().st_size > max_bytes:
            raise MediaError("Decoded audio exceeds the size limit.")
        with wave.open(str(destination), "rb") as audio:
            if (
                audio.getnchannels() != 1
                or audio.getframerate() != _SAMPLE_RATE
                or audio.getsampwidth() != 2
                or audio.getcomptype() != "NONE"
            ):
                raise MediaError("Decoded audio has an unexpected format.")
            expected_frames = audio.getnframes()
            decoded_bytes = 0
            while chunk := audio.readframes(_SAMPLE_RATE):
                decoded_bytes += len(chunk)
            if decoded_bytes != expected_frames * 2:
                raise MediaError("Decoded audio is incomplete.")
            duration = expected_frames / _SAMPLE_RATE
            if duration <= 0:
                raise MediaError("Decoded audio is empty.")
            return duration
    except (OSError, EOFError, wave.Error) as exc:
        raise MediaError("Decoded audio is not a valid WAV file.") from exc


async def normalize_audio(source: Path, destination: Path, max_duration_seconds: int) -> float:
    """Write mono 16 kHz PCM WAV and return its verified duration in seconds.

    The caller owns the temporary directory. A failed or cancelled conversion
    removes its partial output. Source and destination must be different files.
    """
    source, destination = Path(source), Path(destination)
    if (
        isinstance(max_duration_seconds, bool)
        or not isinstance(max_duration_seconds, int)
        or max_duration_seconds <= 0
    ):
        raise ValueError("max_duration_seconds must be a positive integer")
    if source.resolve() == destination.resolve():
        raise MediaError("Source and destination must be different files.")
    if not source.is_file():
        raise MediaError("Source media does not exist.")
    duration = await _probe_duration(source)
    if duration > max_duration_seconds:
        raise MediaError("Media exceeds the maximum permitted duration.")
    max_bytes = (max_duration_seconds + 1) * _BYTES_PER_SECOND + _WAV_HEADER_ALLOWANCE
    try:
        await _run(
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            "-fs",
            str(max_bytes),
            "-f",
            "wav",
            str(destination),
        )
        normalized_duration = _validate_wav(destination, max_bytes)
        if normalized_duration > max_duration_seconds:
            raise MediaError("Decoded audio exceeds the maximum permitted duration.")
        if abs(normalized_duration - duration) > _DURATION_TOLERANCE_SECONDS:
            raise MediaError("Decoded audio duration differs from the source; it may be truncated.")
        return normalized_duration
    except BaseException:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        raise
