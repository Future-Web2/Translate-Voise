from __future__ import annotations

import argparse
import asyncio
import logging
import shutil

from aiogram import Bot
from aiohttp import web

from voicebot.ai import AIClient, AIError
from voicebot.asr import LocalASR
from voicebot.bot import JobRunner, create_dispatcher
from voicebot.config import Settings


async def serve(settings: Settings) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise ValueError("Install ffmpeg and ffprobe (included in the Docker image)")
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    ai = AIClient(settings)
    asr = LocalASR(settings)
    bot = Bot(settings.telegram_token)
    jobs = JobRunner(bot, settings, ai, asr)
    dispatcher = create_dispatcher(settings, jobs)
    alive = False

    async def health(_request: web.Request) -> web.Response:
        healthy = alive and jobs.task is not None and not jobs.task.done()
        return web.json_response(
            {"status": "ok" if healthy else "starting", "asr": asr.state},
            status=200 if healthy else 503,
        )

    app = web.Application()
    app.router.add_get("/health", health)
    server = web.AppRunner(app, access_log=None)
    try:
        # Preserve pending updates; private access checks still apply to every message.
        await bot.delete_webhook(drop_pending_updates=False)
        await bot.set_my_commands(
            [
                {"command": "start", "description": "Как пользоваться ботом"},
                {"command": "ask", "description": "Задать вопрос AI"},
                {"command": "translate", "description": "Перевести текст на узбекский"},
                {"command": "id", "description": "Мой Telegram ID"},
                {"command": "status", "description": "Очередь и состояние"},
                {"command": "help", "description": "Помощь"},
            ]
        )
        await server.setup()
        await web.TCPSite(server, "0.0.0.0", settings.port).start()
        jobs.start()
        alive = True
        if not settings.allowed_users:
            logging.warning("ALLOWED_USER_IDS is empty; only /id is accessible")
        await dispatcher.start_polling(
            bot,
            allowed_updates=["message"],
            tasks_concurrency_limit=16,
            close_bot_session=False,
        )
    finally:
        alive = False
        await jobs.close()
        await server.cleanup()
        await ai.close()
        await bot.session.close()


async def list_models(settings: Settings) -> None:
    client = AIClient(settings)
    try:
        for model in await client.models():
            print(model)
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Telegram voice transcription and Uzbek translation"
    )
    parser.add_argument("--models", action="store_true", help="List io.net model IDs")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    try:
        settings = Settings.from_env()
        asyncio.run(list_models(settings) if args.models else serve(settings))
    except (ValueError, AIError) as exc:
        parser.exit(1, f"Configuration/API error: {exc}\n")


if __name__ == "__main__":
    main()
