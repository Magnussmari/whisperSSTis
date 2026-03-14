"""Tests for the GPT post-processing module."""

import os

import pytest
from unittest.mock import MagicMock, patch

from whisperSSTis import gpt


class TestGPTConfig:
    """Tests for GPTConfig defaults and overrides."""

    def test_default_values(self):
        cfg = gpt.GPTConfig()
        assert cfg.temperature == 0.3
        assert cfg.max_tokens == 400
        assert cfg.api_key is None
        assert cfg.base_url is None

    def test_custom_values(self):
        cfg = gpt.GPTConfig(api_key="test-key", temperature=0.7, max_tokens=800)
        assert cfg.api_key == "test-key"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 800


class TestBuildClient:
    """Tests for _build_client with various configurations."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=False)
    @patch("whisperSSTis.gpt.OpenAI")
    def test_uses_env_api_key(self, mock_openai_cls):
        cfg = gpt.GPTConfig()
        gpt._build_client(cfg)
        mock_openai_cls.assert_called_once_with(api_key="env-key")

    @patch("whisperSSTis.gpt.OpenAI")
    def test_uses_config_api_key(self, mock_openai_cls):
        cfg = gpt.GPTConfig(api_key="direct-key")
        gpt._build_client(cfg)
        mock_openai_cls.assert_called_once_with(api_key="direct-key")

    @patch("whisperSSTis.gpt.OpenAI")
    def test_includes_base_url_when_set(self, mock_openai_cls):
        cfg = gpt.GPTConfig(api_key="key", base_url="https://custom.api/v1")
        gpt._build_client(cfg)
        mock_openai_cls.assert_called_once_with(
            api_key="key", base_url="https://custom.api/v1"
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_raises_without_api_key(self):
        cfg = gpt.GPTConfig()
        with pytest.raises(RuntimeError, match="Missing OPENAI_API_KEY"):
            gpt._build_client(cfg)

    def test_raises_when_openai_not_installed(self, monkeypatch):
        monkeypatch.setattr(gpt, "OpenAI", None)
        cfg = gpt.GPTConfig(api_key="key")
        with pytest.raises(RuntimeError, match="not installed"):
            gpt._build_client(cfg)


class TestRunOnTranscript:
    """Tests for run_on_transcript end-to-end."""

    @patch("whisperSSTis.gpt._build_client")
    def test_returns_gpt_response(self, mock_build):
        mock_client = MagicMock()
        mock_build.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary of transcript"
        mock_client.chat.completions.create.return_value = mock_response

        result = gpt.run_on_transcript(
            transcript="Halló heimur",
            instruction="Summarize",
            config=gpt.GPTConfig(api_key="test"),
        )

        assert result == "Summary of transcript"
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 400
        assert len(call_kwargs["messages"]) == 2
        assert "Halló heimur" in call_kwargs["messages"][1]["content"]

    @patch("whisperSSTis.gpt._build_client")
    def test_returns_empty_on_none_content(self, mock_build):
        mock_client = MagicMock()
        mock_build.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response

        result = gpt.run_on_transcript(
            transcript="test",
            instruction="test",
            config=gpt.GPTConfig(api_key="test"),
        )
        assert result == ""

    @patch("whisperSSTis.gpt._build_client")
    def test_uses_default_config_when_none(self, mock_build):
        mock_client = MagicMock()
        mock_build.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_client.chat.completions.create.return_value = mock_response

        gpt.run_on_transcript("text", "instruction")
        mock_build.assert_called_once()

    @patch("whisperSSTis.gpt._build_client")
    def test_respects_custom_temperature_and_tokens(self, mock_build):
        mock_client = MagicMock()
        mock_build.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_client.chat.completions.create.return_value = mock_response

        cfg = gpt.GPTConfig(api_key="key", temperature=0.9, max_tokens=1200)
        gpt.run_on_transcript("text", "instruction", config=cfg)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.9
        assert call_kwargs["max_tokens"] == 1200
