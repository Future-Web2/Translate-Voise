from __future__ import annotations

import asyncio
import contextlib
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramRetryAfter
from aiogram.filters import Command, CommandStart
from aiogram.types import BufferedInputFile, Message

from voicebot.ai import AIClient, AIError
from voicebot.asr import ASRError, LocalASR
from voicebot.config import Settings
from voicebot.media import MediaError, normalize_audio
from voicebot.text import build_transcript, split_text

logger = logging.getLogger(__name__)

HELP = """Отправьте голосовое сообщение или видеокружок.
Я пришлю расшифровку на исходных языках, а ниже — перевод на узбекский латиницей.

/ask вопрос — отдельный ответ AI; обычный текст тоже считается вопросом.
/translate текст — перевод текста на узбекский.
/id — ваш Telegram ID.
/status — состояние очереди.

Речь может смешивать узбекский, русский и английский. Метки [неуверенно ...]
показывают сомнительные места; точность нужно проверять по записи.

Аудио обрабатывается локально на сервере. Расшифровка для перевода и вопросы
отправляются в io.net. Временные файлы удаляются после обработки.
История переписки на сервере не сохраняется. В Telegram ваши сообщения остаются."""


async def send_text(message: Message, text: str) -> None:
    for attempt in range(3):
        try:
            await message.answer(text, parse_mode=None)
            return
        except TelegramRetryAfter as exc:
            if attempt == 2 or exc.retry_after > 30:
                raise
            await asyncio.sleep(exc.retry_after + 0.1)


async def send_result(message: Message, text: str, filename: str) -> None:
    chunks = split_text(text)
    if len(chunks) > 4:
        await send_text(message, chunks[0] + "\n\nПолный текст — в приложенном файле.")
        await message.answer_document(
            BufferedInputFile(text.encode("utf-8"), filename=filename),
            caption="Полный результат (UTF-8)",
            parse_mode=None,
        )
    else:
        for chunk in chunks:
            await send_text(message, chunk)


class LimitedWriter:
    """Bot.download_file writes incrementally; enforce actual bytes, not just metadata."""

    def __init__(self, stream, limit: int):
        self.stream = stream
        self.limit = limit
        self.size = 0

    def write(self, chunk: bytes) -> int:
        self.size += len(chunk)
        if self.size > self.limit:
            raise MediaError("Файл превышает допустимый размер.")
        return self.stream.write(chunk)

    def flush(self) -> None:
        self.stream.flush()


@dataclass(frozen=True)
class Job:
    message: Message
    kind: str
    text: str = ""


class JobRunner:
    def __init__(self, bot: Bot, settings: Settings, ai: AIClient, asr: LocalASR):
        self.bot, self.settings, self.ai, self.asr = bot, settings, ai, asr
        self.queue: asyncio.Queue[Job] = asyncio.Queue(maxsize=settings.queue_size)
        self.busy: set[int] = set()
        self.last_submitted: dict[int, float] = {}
        self.task: asyncio.Task | None = None
        self.accepting = True

    def start(self) -> None:
        self.task = asyncio.create_task(self.run(), name="job-runner")

    async def close(self) -> None:
        self.accepting = False
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
        await self.asr.close()

    async def submit(self, job: Job) -> None:
        user_id = job.message.from_user.id
        now = time.monotonic()
        if not self.accepting:
            await send_text(job.message, "Бот перезапускается. Отправьте сообщение чуть позже.")
        elif user_id in self.busy:
            await send_text(job.message, "Ваш предыдущий запрос ещё обрабатывается.")
        elif now - self.last_submitted.get(user_id, -float("inf")) < self.settings.cooldown:
            await send_text(job.message, "Подождите несколько секунд между запросами.")
        elif self.queue.full():
            await send_text(job.message, "Очередь заполнена. Попробуйте чуть позже.")
        else:
            # No await before put/busy: admission remains atomic on the event loop.
            self.busy.add(user_id)
            self.last_submitted[user_id] = now
            self.queue.put_nowait(job)

    async def run(self) -> None:
        while True:
            job = await self.queue.get()
            try:
                await send_text(job.message, "Обрабатываю запрос…")
                if job.kind == "media":
                    await self.process_media(job.message)
                elif job.kind == "translate":
                    translated = await self.ai.translate(job.text)
                    await send_result(
                        job.message, f"O‘zbekcha tarjima:\n{translated}", "tarjima.txt"
                    )
                else:
                    answer = await self.ai.answer(job.text)
                    await send_result(job.message, answer, "answer.txt")
            except (AIError, ASRError) as exc:
                await self.report_error(job.message, str(exc))
            except MediaError:
                await self.report_error(
                    job.message,
                    "Не удалось обработать аудио. Проверьте размер, длительность и наличие звука.",
                )
            except Exception as exc:
                # Never log message text, bot token, provider body or media path.
                logger.error("Job failed (%s)", type(exc).__name__)
                await self.report_error(
                    job.message, "Не удалось завершить запрос. Отправьте его ещё раз."
                )
            finally:
                self.busy.discard(job.message.from_user.id)
                self.queue.task_done()

    async def report_error(self, message: Message, text: str) -> None:
        try:
            await send_text(message, text)
        except Exception as exc:
            logger.warning("Cannot send error (%s)", type(exc).__name__)

    async def process_media(self, message: Message) -> None:
        media = message.voice or message.video_note
        if media is None:
            raise MediaError("Unsupported media")
        if media.duration > self.settings.max_duration:
            raise MediaError("Duration exceeded")
        if media.file_size and media.file_size > self.settings.max_file_bytes:
            raise MediaError("File too large")
        with tempfile.TemporaryDirectory(prefix="voicebot-") as directory:
            source = Path(directory) / "source.media"
            normalized = Path(directory) / "audio.wav"
            remote = await self.bot.get_file(media.file_id)
            if not remote.file_path:
                raise MediaError("Missing file path")
            if remote.file_size and remote.file_size > self.settings.max_file_bytes:
                raise MediaError("File too large")
            with source.open("wb") as output:
                destination = LimitedWriter(output, self.settings.max_file_bytes)
                url = self.bot.session.api.file_url(self.bot.token, remote.file_path)
                # Close HTTP streaming promptly even if the limit is exceeded or cancelled.
                async with contextlib.aclosing(
                    self.bot.session.stream_content(
                        url=url,
                        timeout=90,
                        chunk_size=65536,
                        raise_for_status=True,
                    )
                ) as stream:
                    async for chunk in stream:
                        destination.write(chunk)
            await normalize_audio(source, normalized, self.settings.max_duration)
            segments = await self.asr.transcribe(normalized)
            transcript = build_transcript(segments)
            if not transcript.text:
                await send_text(
                    message, "Не удалось распознать речь. Попробуйте более чёткую запись."
                )
                return
            try:
                translation = await self.ai.translate(transcript.text)
            except AIError as exc:
                # Preserve successful transcription when the text provider is unavailable.
                await send_result(
                    message,
                    f"Расшифровка:\n{transcript.text}\n\nПеревод недоступен: {exc}",
                    "transcript.txt",
                )
                return
            result = f"Расшифровка:\n{transcript.text}\n\nO‘zbekcha tarjima:\n{translation}"
            if transcript.uncertain_ranges:
                result += (
                    "\n\nМетки «неуверенно» — сигнал перепроверить запись, "
                    "а не точная оценка ошибок."
                )
            await send_result(message, result, "transcript-uz.txt")


def create_dispatcher(settings: Settings, runner: JobRunner) -> Dispatcher:
    router = Router()

    async def authorized(message: Message) -> bool:
        if message.chat.type != "private":
            return False
        if not message.from_user or message.from_user.id not in settings.allowed_users:
            await send_text(
                message, "Доступ закрыт. Узнайте свой ID командой /id и передайте владельцу."
            )
            return False
        return True

    @router.message(Command("id"))
    async def identity(message: Message) -> None:
        if message.chat.type == "private" and message.from_user:
            await send_text(message, f"Ваш Telegram user ID: {message.from_user.id}")

    @router.message(CommandStart())
    @router.message(Command("help"))
    async def help_message(message: Message) -> None:
        if await authorized(message):
            await send_text(message, HELP)

    @router.message(Command("status"))
    async def status(message: Message) -> None:
        if await authorized(message):
            await send_text(
                message,
                f"Запросов в очереди: {runner.queue.qsize()}. "
                f"Распознавание: {runner.asr.state}.\n"
                "Первый запрос загружает модель и может занять несколько минут.",
            )

    @router.message(F.voice | F.video_note)
    async def media_message(message: Message) -> None:
        if await authorized(message):
            media = message.voice or message.video_note
            if media.duration > settings.max_duration:
                await send_text(
                    message, f"Максимальная длительность: {settings.max_duration} секунд."
                )
            elif media.file_size and media.file_size > settings.max_file_bytes:
                await send_text(
                    message, f"Максимальный размер: {settings.max_file_bytes // 1048576} МБ."
                )
            else:
                await runner.submit(Job(message, "media"))

    @router.message(Command("ask", "translate"))
    async def command_message(message: Message) -> None:
        if await authorized(message):
            parts = message.text.split(maxsplit=1)
            command = parts[0]
            text = parts[1].strip() if len(parts) == 2 else ""
            kind = "translate" if command.split("@")[0] == "/translate" else "qa"
            if not text:
                await send_text(message, f"Укажите текст после {command}.")
            else:
                await runner.submit(Job(message, kind, text))

    @router.message(F.text)
    async def text_message(message: Message) -> None:
        if await authorized(message):
            if message.text.startswith("/"):
                await send_text(message, "Неизвестная команда. Доступные команды: /help.")
            else:
                await runner.submit(Job(message, "qa", message.text))

    @router.message()
    async def unsupported_message(message: Message) -> None:
        if await authorized(message):
            await send_text(message, "Отправьте голосовое, видеокружок или текстовый вопрос.")

    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    return dispatcher
