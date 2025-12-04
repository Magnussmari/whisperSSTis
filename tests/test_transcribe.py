import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from unittest.mock import patch
from whisperSSTis import transcribe
import numpy as np

def test_format_timestamp():
    """Test format_timestamp function."""
    seconds = 123.456
    timestamp = transcribe.format_timestamp(seconds)
    assert timestamp == "0:02:03"

def test_create_srt():
    """Test create_srt function."""
    transcriptions = [
        "[0:00:00 → 0:00:05] Hello world",
        "[0:00:05 → 0:00:10] This is a test"
    ]
    srt_content = transcribe.create_srt(transcriptions)
    assert "1" in srt_content
    assert "0:00:00,000 --> 0:00:05,000" in srt_content
    assert "Hello world" in srt_content
    assert "2" in srt_content
    assert "0:00:05,000 --> 0:00:10,000" in srt_content
    assert "This is a test" in srt_content

@patch('whisperSSTis.transcribe.WhisperProcessor.from_pretrained')
@patch('whisperSSTis.transcribe.WhisperForConditionalGeneration.from_pretrained')
def test_load_model(mock_processor, mock_model):
    """Test load_model function."""
    model, processor = transcribe.load_model()
    assert model is not None
    assert processor is not None

def test_transcribe_audio(mocker):
    """Test transcribe_audio function."""
    # Mock the model and processor
    mock_model = mocker.MagicMock()
    mock_processor = mocker.MagicMock()

    # Mock the processor's return value
    mock_processor_output = mocker.MagicMock()
    mock_processor.return_value = mock_processor_output
    mock_processor_output.input_features.to.return_value = "mocked_input_features"

    # Mock the model's generate method
    mock_model.generate.return_value = [[1, 2, 3]]

    # Mock the processor's decode method
    mock_processor.decode.return_value = "Mocked transcription"

    # Create mock audio data
    mock_audio_data = np.zeros(16000, dtype=np.float32)

    # Call the function
    transcription = transcribe.transcribe_audio(mock_audio_data, mock_model, mock_processor)

    # Assertions
    assert transcription == "Mocked transcription"

@patch('whisperSSTis.transcribe.WhisperProcessor.from_pretrained')
@patch('whisperSSTis.transcribe.WhisperForConditionalGeneration.from_pretrained')
def test_transcribe_audio_exception(mock_processor, mock_model, mocker):
    """Test transcribe_audio function with exception."""
    # Mock the model and processor
    mock_model = mocker.MagicMock()
    mock_processor = mocker.MagicMock()

    # Mock the processor's return value
    mock_processor_output = mocker.MagicMock()
    mock_processor.return_value = mock_processor_output
    mock_processor_output.input_features.to.return_value = "mocked_input_features"

    # Mock the model's generate method to raise an exception
    mock_model.generate.side_effect = Exception("Test exception")

    # Create mock audio data
    mock_audio_data = np.zeros(16000, dtype=np.float32)

    # Call the function and assert that it raises an exception
    with pytest.raises(Exception, match="Test exception"):
        transcribe.transcribe_audio(mock_audio_data, mock_model, mock_processor)

@patch('whisperSSTis.transcribe.WhisperProcessor.from_pretrained')
@patch('whisperSSTis.transcribe.WhisperForConditionalGeneration.from_pretrained')
def test_transcribe_long_audio(mock_processor, mock_model, mocker):
    """Test transcribe_long_audio function."""
    # Mock the model and processor
    mock_model_instance = mock_model.return_value
    mock_processor_instance = mock_processor.return_value

    # Mock the processor's return value
    mock_processor_output = mocker.MagicMock()
    mock_processor_instance.return_value = mock_processor_output
    mock_processor_output.input_features.to.return_value = "mocked_input_features"

    # Mock the model's generate method
    mock_model_instance.generate.return_value = [[1, 2, 3]]

    # Mock the processor's decode method
    mock_processor_instance.decode.return_value = "Mocked transcription"

    # Create mock audio data for three seconds
    mock_audio_data = np.zeros(48000, dtype=np.float32)

    # Call the function with one-second chunks
    transcriptions = transcribe.transcribe_long_audio(
        mock_audio_data,
        mock_model_instance,
        mock_processor_instance,
        duration=3,
        chunk_size=1,
    )

    # Assertions
    assert isinstance(transcriptions, list)
    assert len(transcriptions) == 3
    assert transcriptions[0] == "[0:00:00 → 0:00:01] Mocked transcription"
    assert transcriptions[-1] == "[0:00:02 → 0:00:03] Mocked transcription"
