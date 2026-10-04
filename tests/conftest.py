import pytest

from voicebot.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(
        telegram_token="123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi",
        translation_key="translation-secret",
        qa_key="qa-secret",
        allowed_users=frozenset({123}),
        translation_model="translation-model",
        qa_model="qa-model",
        cache_dir=tmp_path / "models",
        cooldown=0,
    )
