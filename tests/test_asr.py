from __future__ import annotations

import asyncio
import io
import json
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from voicebot import asr, asr_worker
from voicebot.config import Settings


def settings(**overrides):
    base = Settings(
        telegram_token="private-token",
        translation_key="private-translation",
        qa_key="private-qa",
        allowed_users=frozenset({1}),
        translation_model="test",
        qa_model="test",
    )
    return replace(base, **overrides)


SEGMENTS = [
    {
        "start": 0.0,
        "end": 1.0,
        "text": "Salom world",
        "avg_logprob": -0.1,
        "words": [{"start": 0.0, "end": 0.5, "word": "Salom", "probability": 0.9}],
    }
]


def process_with_response(response=None):
    return SimpleNamespace(
        returncode=None,
        stdin=SimpleNamespace(write=Mock(), drain=AsyncMock()),
        stdout=SimpleNamespace(
            readline=AsyncMock(
                return_value=json.dumps(response or {"ok": True, "segments": SEGMENTS}).encode()
                + b"\n"
            )
        ),
        kill=Mock(),
        wait=AsyncMock(return_value=0),
    )


def test_process_is_reused_and_request_omits_secrets(tmp_path):
    process = process_with_response()
    spawn = AsyncMock(return_value=process)

    async def run():
        service = asr.LocalASR(settings(cache_dir=tmp_path / "models"))
        assert service.state == "cold"
        assert await service.transcribe(tmp_path / "audio.wav") == SEGMENTS
        assert service.state == "ready"
        await service.transcribe(tmp_path / "audio.wav")
        await service.close()
        assert service.state == "cold"
        with pytest.raises(asr.ASRError):
            await service.transcribe(tmp_path / "audio.wav")

    with patch.object(asr.asyncio, "create_subprocess_exec", spawn):
        asyncio.run(run())
    spawn.assert_awaited_once()
    args, kwargs = spawn.call_args
    assert args == (sys.executable, "-m", "voicebot.asr_worker")
    assert kwargs["limit"] == 1024 * 1024
    raw = process.stdin.write.call_args.args[0]
    assert b"private-" not in raw
    payload = json.loads(raw)
    assert payload["settings"]["model"] == "large-v3"
    assert payload["settings"]["language"] is None
    process.kill.assert_called_once()
    process.wait.assert_awaited_once()


def test_timeout_kills_reaps_and_next_job_restarts(tmp_path):
    stalled = process_with_response()
    healthy = process_with_response()

    async def hang():
        await asyncio.Event().wait()

    stalled.stdout.readline.side_effect = hang
    spawn = AsyncMock(side_effect=[stalled, healthy])

    async def run():
        service = asr.LocalASR(settings(asr_timeout=0.02))
        with pytest.raises(asr.ASRError, match="слишком много"):
            await service.transcribe(tmp_path / "audio.wav")
        assert service.state == "cold"
        stalled.kill.assert_called_once()
        stalled.wait.assert_awaited_once()
        assert await service.transcribe(tmp_path / "audio.wav") == SEGMENTS
        await service.close()

    with patch.object(asr.asyncio, "create_subprocess_exec", spawn):
        asyncio.run(run())
    assert spawn.await_count == 2


def test_cancelled_job_kills_and_reaps(tmp_path):
    process = process_with_response()

    async def run():
        entered = asyncio.Event()

        async def hang():
            entered.set()
            await asyncio.Event().wait()

        process.stdout.readline.side_effect = hang
        service = asr.LocalASR(settings())
        job = asyncio.create_task(service.transcribe(tmp_path / "audio.wav"))
        await entered.wait()
        assert service.state == "working"
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        assert service.state == "cold"
        process.kill.assert_called_once()
        process.wait.assert_awaited_once()

    with patch.object(asr.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)):
        asyncio.run(run())


def test_cancelling_queued_job_does_not_kill_running_job(tmp_path):
    process = process_with_response()

    async def run():
        entered, finish = asyncio.Event(), asyncio.Event()

        async def delayed():
            entered.set()
            await finish.wait()
            return json.dumps({"ok": True, "segments": SEGMENTS}).encode() + b"\n"

        process.stdout.readline.side_effect = delayed
        service = asr.LocalASR(settings())
        first = asyncio.create_task(service.transcribe(tmp_path / "first.wav"))
        await entered.wait()
        queued = asyncio.create_task(service.transcribe(tmp_path / "second.wav"))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        process.kill.assert_not_called()
        finish.set()
        assert await first == SEGMENTS
        await service.close()

    with patch.object(asr.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)):
        asyncio.run(run())


@pytest.mark.parametrize(
    "response",
    [
        {"ok": False, "error": "private transcript"},
        {"ok": True, "segments": "private transcript"},
        {"ok": True, "segments": [{**SEGMENTS[0], "avg_logprob": float("nan")}]},
    ],
)
def test_invalid_worker_response_is_safe_and_process_is_reaped(tmp_path, response):
    process = process_with_response(response)

    async def run():
        with pytest.raises(asr.ASRError) as raised:
            await asr.LocalASR(settings()).transcribe(tmp_path / "audio.wav")
        assert "private" not in str(raised.value)

    with patch.object(asr.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)):
        asyncio.run(run())
    process.kill.assert_called_once()
    process.wait.assert_awaited_once()


def test_worker_caches_model_and_forwards_multilingual_options(monkeypatch):
    model = Mock()
    word = SimpleNamespace(start=0, end=1, word="Salom", probability=0.8)
    segment = SimpleNamespace(start=0, end=1, text="Salom", avg_logprob=-0.2, words=[word])
    model.transcribe.side_effect = [([segment], None), ([segment], None)]
    constructor = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=constructor))
    request_settings = {
        "model": "large-v3",
        "cpu_threads": 2,
        "cache_dir": "models",
        "language": None,
        "glossary": "Railway, Toshkent",
    }
    requests = [
        {"path": "first.wav", "settings": request_settings},
        {"path": "second.wav", "settings": {**request_settings, "language": "uz"}},
    ]
    stdin = io.TextIOWrapper(
        io.BytesIO(b"".join((json.dumps(request) + "\n").encode() for request in requests))
    )
    stdout = io.TextIOWrapper(io.BytesIO())
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    asr_worker.main()
    stdout.flush()
    responses = [json.loads(line) for line in stdout.buffer.getvalue().splitlines()]
    assert len(responses) == 2
    assert all(response["ok"] is True for response in responses)
    constructor.assert_called_once_with(
        "large-v3",
        device="cpu",
        compute_type="int8",
        cpu_threads=2,
        num_workers=1,
        download_root="models",
    )
    first, second = model.transcribe.call_args_list
    assert first.kwargs["multilingual"] is True
    assert second.kwargs["multilingual"] is False
    assert first.kwargs["condition_on_previous_text"] is False
    assert first.kwargs["word_timestamps"] is True
    assert first.kwargs["hotwords"] == "Railway, Toshkent"
