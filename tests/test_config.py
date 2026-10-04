import pytest

from voicebot.config import Settings


def set_environment(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:secret")
    monkeypatch.setenv("IO_TRANSLATION_API_KEY", "translate-secret")
    monkeypatch.setenv("IO_QA_API_KEY", "qa-secret")
    monkeypatch.setenv("IO_TRANSLATION_MODEL", "model")
    monkeypatch.setenv("IO_QA_MODEL", "model")


def test_empty_allowlist_denies_everyone_by_default(monkeypatch):
    set_environment(monkeypatch)
    monkeypatch.delenv("ALLOWED_USER_IDS", raising=False)
    settings = Settings.from_env()
    assert settings.allowed_users == frozenset()
    assert "secret" not in repr(settings)


@pytest.mark.parametrize(
    "name,value",
    [
        ("MAX_FILE_MB", "21"),
        ("MAX_DURATION_SECONDS", "0"),
        ("ALLOWED_USER_IDS", "abc"),
        ("ALLOWED_USER_IDS", "-1"),
        ("WHISPER_MODEL", "small.en"),
        ("ASR_LANGUAGE", "nonsense"),
        ("IO_BASE_URL", "http://unsafe.example"),
    ],
)
def test_invalid_settings_fail_at_start(monkeypatch, name, value):
    set_environment(monkeypatch)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        Settings.from_env()
