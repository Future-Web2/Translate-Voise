import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from voicebot.ai import AIClient, AIError


def completion(content="Salom", reason="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": reason}]}


async def test_separate_credentials_models_and_translation_data(settings):
    calls = []

    def handle(request):
        calls.append((request.headers["authorization"], json.loads(request.content)))
        return httpx.Response(200, json=completion())

    async with httpx.AsyncClient(
        base_url=settings.io_base_url + "/",
        transport=httpx.MockTransport(handle),
    ) as http:
        ai = AIClient(settings, http)
        await ai.translate("Salom, hello. [неуверенно 00:01–00:02: test]")
        await ai.answer("What is Python?")
    assert calls[0][0] == "Bearer translation-secret"
    assert calls[0][1]["model"] == "translation-model"
    assert json.loads(calls[0][1]["messages"][1]["content"])["transcript"].startswith("Salom")
    assert calls[1][0] == "Bearer qa-secret"
    assert calls[1][1]["model"] == "qa-model"
    assert calls[1][1]["messages"][1]["content"] == "What is Python?"


@pytest.mark.parametrize("status", [401, 402, 403, 400])
async def test_provider_errors_do_not_leak_body_or_key(settings, status):
    async with httpx.AsyncClient(
        base_url=settings.io_base_url + "/",
        transport=httpx.MockTransport(lambda _: httpx.Response(status, text="secret user data")),
    ) as http:
        with pytest.raises(AIError) as error:
            await AIClient(settings, http).answer("question")
    assert "secret" not in str(error.value)


async def test_rate_limit_has_bounded_retries(settings):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(429, headers={"Retry-After": "500"})

    async with httpx.AsyncClient(
        base_url=settings.io_base_url + "/",
        transport=httpx.MockTransport(handle),
    ) as http:
        with patch("voicebot.ai.asyncio.sleep", new_callable=AsyncMock) as sleep:
            with pytest.raises(AIError):
                await AIClient(settings, http).answer("question")
            assert sleep.await_count == 2
            assert all(call.args[0] < 21 for call in sleep.await_args_list)
    assert len(requests) == 3


async def test_truncated_translation_never_reported_complete(settings):
    async with httpx.AsyncClient(
        base_url=settings.io_base_url + "/",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=completion(reason="length"))
        ),
    ) as http:
        with pytest.raises(AIError, match="обрезал"):
            await AIClient(settings, http).translate("text")


async def test_catalog_supports_both_formats_and_pagination(settings):
    def handle(request):
        if request.url.params["page"] == "1":
            return httpx.Response(
                200,
                json={
                    "data": [{"id": "old/model"}],
                    "pagination": {"total_pages": 2},
                },
            )
        return httpx.Response(
            200,
            json={
                "data": [{"name": "new/model", "model_id": "uuid"}],
                "pagination": {"total_pages": 2},
            },
        )

    async with httpx.AsyncClient(
        base_url=settings.io_base_url + "/",
        transport=httpx.MockTransport(handle),
    ) as http:
        assert await AIClient(settings, http).models() == ["old/model", "new/model"]
