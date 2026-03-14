"""Shared test fixtures for whisperSSTis test suite."""

import numpy as np
import pytest


@pytest.fixture
def silence_1s():
    """One second of silence at 16kHz as float32."""
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def silence_3s():
    """Three seconds of silence at 16kHz as float32."""
    return np.zeros(48000, dtype=np.float32)


@pytest.fixture
def mock_model(mocker):
    """A mocked Whisper model that returns predictable token IDs."""
    model = mocker.MagicMock()
    model.generate.return_value = [[1, 2, 3]]
    return model


@pytest.fixture
def mock_processor(mocker):
    """A mocked WhisperProcessor with working __call__ and decode."""
    processor = mocker.MagicMock()
    processor_output = mocker.MagicMock()
    processor.return_value = processor_output
    processor_output.input_features.to.return_value = "mocked_input_features"
    processor.decode.return_value = "Mocked transcription"
    return processor
