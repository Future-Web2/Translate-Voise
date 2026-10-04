import asyncio
import datetime as dt
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.types import Chat, Message, Update, User, Voice

from voicebot.ai import AIError
from voicebot.bot import Job, JobRunner, LimitedWriter, create_dispatcher
from voicebot.media import MediaError


class RecordingSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.requests = []

    async def close(self):
        pass

    async def make_request(self, bot, method, timeout=None):  # noqa: ASYNC109 - BaseSession API
        self.requests.append(method)
        return Message(
            message_id=100,
            date=dt.datetime.now(dt.UTC),
            chat=Chat(id=method.chat_id, type="private"),
            text=getattr(method, "text", ""),
        )

    async def stream_content(self, *args, **kwargs):
        yield b""


def make_message(user_id=123, text=None, voice=None, chat_type="private"):
    return Message(
        message_id=1,
        date=dt.datetime.now(dt.UTC),
        chat=Chat(id=user_id, type=chat_type),
        from_user=User(id=user_id, is_bot=False, first_name="Test"),
        text=text,
        voice=voice,
    )


@pytest.mark.parametrize("kind", ["question", "media", "translate"])
async def test_unauthorized_users_never_reach_ai_or_media(settings, kind):
    session = RecordingSession()
    bot = Bot(settings.telegram_token, session=session)
    runner = SimpleNamespace(submit=AsyncMock())
    dp = create_dispatcher(settings, runner)
    message = make_message(
        user_id=999,
        text="/translate test" if kind == "translate" else "Hello?",
    )
    if kind == "media":
        message = make_message(
            user_id=999,
            voice=Voice(
                file_id="file",
                file_unique_id="unique",
                duration=1,
            ),
        )
    await dp.feed_update(bot, Update(update_id=1, message=message))
    runner.submit.assert_not_awaited()
    assert "Доступ закрыт" in session.requests[0].text


async def test_id_available_before_authorization_and_groups_ignored(settings):
    session = RecordingSession()
    bot = Bot(settings.telegram_token, session=session)
    runner = SimpleNamespace(submit=AsyncMock())
    dp = create_dispatcher(settings, runner)
    await dp.feed_update(bot, Update(update_id=1, message=make_message(999, text="/id")))
    assert "999" in session.requests[0].text
    await dp.feed_update(
        bot,
        Update(
            update_id=2,
            message=make_message(text="private question", chat_type="group"),
        ),
    )
    runner.submit.assert_not_awaited()
    assert len(session.requests) == 1


@pytest.mark.parametrize("separator", [" ", "\n", "\t"])
async def test_voice_and_question_use_different_job_types(settings, separator):
    session = RecordingSession()
    bot = Bot(settings.telegram_token, session=session)
    runner = SimpleNamespace(submit=AsyncMock())
    dp = create_dispatcher(settings, runner)
    await dp.feed_update(
        bot, Update(update_id=1, message=make_message(text=f"/ask{separator}Nima?"))
    )
    await dp.feed_update(
        bot,
        Update(
            update_id=2,
            message=make_message(
                voice=Voice(
                    file_id="file",
                    file_unique_id="unique",
                    duration=1,
                )
            ),
        ),
    )
    assert [call.args[0].kind for call in runner.submit.await_args_list] == ["qa", "media"]
    assert runner.submit.await_args_list[0].args[0].text == "Nima?"


def test_download_actual_size_is_enforced():
    output = io.BytesIO()
    writer = LimitedWriter(output, 5)
    writer.write(b"123")
    with pytest.raises(MediaError):
        writer.write(b"456")
    assert output.getvalue() == b"123"


async def test_translation_failure_keeps_transcript_and_cleans_media(settings):
    ai = SimpleNamespace(translate=AsyncMock(side_effect=AIError("Недоступен")))
    asr = SimpleNamespace(
        transcribe=AsyncMock(
            return_value=[
                {
                    "start": 0,
                    "end": 1,
                    "text": "Salom hello",
                    "avg_logprob": -0.1,
                }
            ]
        )
    )
    paths = []

    async def download(**kwargs):
        yield b"media"

    async def normalize(source, target, max_duration):
        paths.extend([source, target])
        target.write_bytes(b"wav")

    bot = SimpleNamespace(
        get_file=AsyncMock(return_value=SimpleNamespace(file_path="remote", file_size=5)),
        token="fake-token",
        session=SimpleNamespace(
            api=SimpleNamespace(file_url=lambda token, path: "https://example.test/file"),
            stream_content=download,
        ),
    )
    message = SimpleNamespace(
        voice=SimpleNamespace(file_id="file", duration=1, file_size=5),
        video_note=None,
    )
    runner = JobRunner(bot, settings, ai, asr)
    with (
        patch("voicebot.bot.normalize_audio", side_effect=normalize),
        patch("voicebot.bot.send_result", new_callable=AsyncMock) as result,
    ):
        await runner.process_media(message)
        assert "Salom hello" in result.await_args.args[1]
        assert "Перевод недоступен" in result.await_args.args[1]
    assert all(not Path(path).exists() for path in paths)  # noqa: ASYNC240 - test tempfiles


async def test_busy_and_queue_bounds(settings):
    runner = JobRunner(None, settings, None, None)
    message = SimpleNamespace(from_user=SimpleNamespace(id=123), answer=AsyncMock())
    await runner.submit(Job(message, "qa", "one"))
    await runner.submit(Job(message, "qa", "two"))
    assert runner.queue.qsize() == 1
    assert "предыдущий" in message.answer.await_args.args[0]


async def test_worker_survives_failed_error_delivery(settings):
    ai = SimpleNamespace(answer=AsyncMock(side_effect=RuntimeError("failure")))
    asr = SimpleNamespace(close=AsyncMock())
    runner = JobRunner(None, settings, ai, asr)
    message = SimpleNamespace(from_user=SimpleNamespace(id=123), answer=AsyncMock())
    runner.start()
    with patch("voicebot.bot.send_text", new_callable=AsyncMock) as send:
        send.side_effect = RuntimeError("cannot send")
        await runner.submit(Job(message, "qa", "one"))
        await asyncio.wait_for(runner.queue.join(), 1)
        assert not runner.task.done()
        assert 123 not in runner.busy
    await runner.close()
