from __future__ import annotations

import asyncio
import json
import shutil
import wave
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from voicebot import media


def write_wav(path: Path, seconds: float, rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"\0\0" * round(seconds * rate))


def probe_result(seconds: object, *, audio: bool = True) -> bytes:
    return json.dumps(
        {
            "streams": [{"codec_type": "audio" if audio else "video"}],
            "format": {"duration": seconds},
        }
    ).encode()


def test_converts_with_bounded_shell_free_commands(tmp_path: Path) -> None:
    source = tmp_path / "video note; arbitrary.mp4"
    source.write_bytes(b"mock media")
    target = tmp_path / "audio.wav"
    calls = []

    async def spawn(*args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "ffmpeg":
            write_wav(target, 2)
        process = AsyncMock()
        process.returncode = 0
        process.communicate.return_value = (
            probe_result("2.0") if args[0] == "ffprobe" else b"",
            b"",
        )
        return process

    with patch.object(media.asyncio, "create_subprocess_exec", side_effect=spawn):
        assert asyncio.run(media.normalize_audio(source, target, 10)) == 2
    assert len(calls) == 2
    assert str(source) in calls[1][0]
    assert "-fs" in calls[1][0]
    assert "-map" in calls[1][0]
    assert "-af" not in calls[1][0]
    assert calls[1][1]["stdin"] == asyncio.subprocess.DEVNULL


@pytest.mark.parametrize(
    "duration,audio",
    [
        ("nan", True),
        ("inf", True),
        ("0", True),
        ("-1", True),
        ("N/A", True),
        ("1", False),
        ("11", True),
    ],
)
def test_rejects_invalid_and_overlong_input(tmp_path, duration, audio) -> None:
    source = tmp_path / "source"
    source.touch()
    run = AsyncMock(return_value=probe_result(duration, audio=audio))
    with patch.object(media, "_run", run), pytest.raises(media.MediaError):
        asyncio.run(media.normalize_audio(source, tmp_path / "out.wav", 10))
    assert run.await_count == 1


def test_prefers_selected_audio_duration_over_video_duration(tmp_path) -> None:
    source = tmp_path / "source"
    source.touch()
    payload = json.dumps(
        {
            "streams": [{"codec_type": "audio", "duration": "2"}],
            "format": {"duration": "100"},
        }
    ).encode()
    with patch.object(media, "_run", AsyncMock(return_value=payload)):
        assert asyncio.run(media._probe_duration(source)) == 2


def test_rejects_silently_truncated_output_and_cleans_up(tmp_path) -> None:
    source, target = tmp_path / "source", tmp_path / "out.wav"
    source.touch()

    async def run(*args):
        if args[0] == "ffprobe":
            return probe_result("10")
        write_wav(target, 3)
        return b""

    with patch.object(media, "_run", side_effect=run), pytest.raises(media.MediaError):
        asyncio.run(media.normalize_audio(source, target, 10))
    assert not target.exists()


def test_wav_validator_rejects_incomplete_data(tmp_path) -> None:
    target = tmp_path / "out.wav"
    write_wav(target, 1)
    target.write_bytes(target.read_bytes()[:-100])
    with pytest.raises(media.MediaError, match="incomplete"):
        media._validate_wav(target, 100_000)


def test_timeout_kills_and_reaps_process() -> None:
    async def scenario():
        process = AsyncMock()
        process.kill = lambda: kills.append(True)

        async def hang():
            await asyncio.Event().wait()

        process.communicate.side_effect = hang
        with (
            patch.object(media.asyncio, "create_subprocess_exec", return_value=process),
            patch.object(media, "_PROCESS_TIMEOUT_SECONDS", 0.001),
            pytest.raises(media.MediaError, match="time limit"),
        ):
            await media._run("ffmpeg")
        process.wait.assert_awaited_once()

    kills = []
    asyncio.run(scenario())
    assert kills == [True]


def test_cancellation_kills_reaps_and_removes_partial_output(tmp_path) -> None:
    source, target = tmp_path / "source", tmp_path / "out.wav"
    source.touch()
    kills = []

    async def scenario():
        started = asyncio.Event()
        process = AsyncMock()
        process.kill = lambda: kills.append(True)

        async def hang():
            target.write_bytes(b"partial output")
            started.set()
            await asyncio.Event().wait()

        process.communicate.side_effect = hang
        with (
            patch.object(media, "_probe_duration", AsyncMock(return_value=1)),
            patch.object(media.asyncio, "create_subprocess_exec", return_value=process),
        ):
            task = asyncio.create_task(media.normalize_audio(source, target, 10))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        process.wait.assert_awaited_once()

    asyncio.run(scenario())
    assert kills == [True]
    assert not target.exists()


@pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="FFmpeg and ffprobe are not installed",
)
def test_real_ffmpeg_smoke(tmp_path) -> None:
    source, target = tmp_path / "source.wav", tmp_path / "normalized.wav"
    write_wav(source, 0.5, rate=48_000)
    assert asyncio.run(media.normalize_audio(source, target, 10)) == pytest.approx(0.5)
    with wave.open(str(target), "rb") as output:
        assert output.getframerate() == 16_000
        assert output.getnchannels() == 1
