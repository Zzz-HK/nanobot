"""ProviderConfig / AppConfig / load_config 的 pytest 测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mini_nanobot.config import (
    AppConfig,
    ConfigurationError,
    ProviderConfig,
    load_config,
)


def _valid_provider_kwargs(**overrides) -> dict:
    data = {
        "api_key": "sk-test",
        "api_base": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 4096,
        "timeout_seconds": 120,
    }
    data.update(overrides)
    return data


@pytest.fixture
def provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    monkeypatch.setenv("MODEL_NAME", "gpt-4o-mini")
    monkeypatch.setenv("MODEL_TEMPERATURE", "0.7")
    monkeypatch.setenv("MODEL_MAX_TOKENS", "4096")
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "120")


class TestProviderConfig:
    def test_valid_explicit_fields(self) -> None:
        cfg = ProviderConfig(**_valid_provider_kwargs())
        assert cfg.api_key == "sk-test"
        assert cfg.api_base == "https://api.openai.com/v1"
        assert cfg.model == "gpt-4o-mini"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096
        assert cfg.timeout_seconds == 120

    def test_reads_from_environment(self, provider_env: None) -> None:
        cfg = ProviderConfig()
        assert cfg.api_key == "sk-test"
        assert cfg.model == "gpt-4o-mini"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096
        assert cfg.timeout_seconds == 120

    def test_strips_trailing_slash_on_api_base(self) -> None:
        cfg = ProviderConfig(
            **_valid_provider_kwargs(api_base="https://api.openai.com/v1/")
        )
        assert cfg.api_base == "https://api.openai.com/v1"

    def test_empty_api_base_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            ProviderConfig(**_valid_provider_kwargs(api_base="   /"))

    @pytest.mark.parametrize(
        "placeholder",
        [
            "https://api.example.com/[workspace-id]/v1",
            "https://api.example.com/<workspace-id>/v1",
            "https://api.example.com/{workspace_id}/v1",
        ],
    )
    def test_placeholder_workspace_id_raises_configuration_error(
        self, placeholder: str
    ) -> None:
        with pytest.raises(ConfigurationError, match="真实值"):
            ProviderConfig(**_valid_provider_kwargs(api_base=placeholder))

    @pytest.mark.parametrize(
        "bad_url",
        [
            "ftp://api.openai.com/v1",
            "not-a-url",
            "https://",
        ],
    )
    def test_invalid_api_base_raises_validation_error(self, bad_url: str) -> None:
        with pytest.raises(ValidationError):
            ProviderConfig(**_valid_provider_kwargs(api_base=bad_url))


class TestAppConfig:
    def test_workspace_paths(self, tmp_path: Path, provider_env: None) -> None:
        cfg = AppConfig(workspace_dir=tmp_path)
        assert cfg.workspace_dir == tmp_path
        assert cfg.db_path == tmp_path / "sessions.db"
        assert cfg.memory_dir == tmp_path / "memory"

    def test_ensure_dirs_creates_workspace_and_memory(
        self, tmp_path: Path, provider_env: None
    ) -> None:
        workspace = tmp_path / "ws"
        cfg = AppConfig(workspace_dir=workspace)
        cfg.ensure_dirs()
        assert workspace.is_dir()
        assert cfg.memory_dir.is_dir()

    def test_default_workspace_from_env(
        self, tmp_path: Path, provider_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "from-env"))
        cfg = AppConfig()
        assert cfg.workspace_dir == (tmp_path / "from-env").resolve()


class TestLoadConfig:
    def test_success_creates_dirs_and_returns_app_config(
        self,
        tmp_path: Path,
        provider_env: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "loaded"))
        monkeypatch.setattr("mini_nanobot.config.load_dotenv", lambda: None)

        cfg = load_config()

        assert isinstance(cfg, AppConfig)
        assert cfg.provider.api_key == "sk-test"
        assert cfg.workspace_dir.is_dir()
        assert cfg.memory_dir.is_dir()

    def test_invalid_env_wraps_as_configuration_error(
        self, provider_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_BASE", "not-a-url")
        monkeypatch.setattr("mini_nanobot.config.load_dotenv", lambda: None)

        with pytest.raises(ConfigurationError, match="配置错误"):
            load_config()
