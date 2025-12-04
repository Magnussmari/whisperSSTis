import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import sounddevice as sd
import numpy as np
from unittest.mock import patch
from whisperSSTis import audio
from scipy import signal

@pytest.fixture
def mock_audio_devices():
    """Mock audio devices for testing."""
    return {
        "Device 1 (ID: 0)": 0,
        "Device 2 (ID: 1)": 1
    }

@patch('whisperSSTis.audio.sd.query_devices')
def test_get_audio_devices(mock_query_devices, mock_audio_devices):
    """Test get_audio_devices function."""
    mock_query_devices.return_value = [{'name': 'Device 1', 'max_input_channels': 1, 'default_samplerate': 48000}]
    devices = audio.get_audio_devices()
    assert "Device 1 (ID: 0)" in devices

@patch('whisperSSTis.audio.sd.query_devices', return_value={'default_samplerate': 48000})
@patch('whisperSSTis.audio.sd.InputStream')
def test_audio_stream_start_stop(mock_input_stream, _query_devices):
    """Test AudioStream start and stop methods."""
    stream = audio.AudioStream(device_id=0, samplerate=16000, chunk_size=1024)
    
    # Mock stream object
    mock_stream = mock_input_stream.return_value
    mock_stream.active = False  # Initially not active
    
    # Start stream
    stream.start_stream()
    mock_input_stream.assert_called_once()
    assert stream.stream is not None
    assert stream.is_recording is True
    
    # Stop stream
    stream.stop_stream()
    assert stream.stream is None
    assert stream.is_recording is False

@patch('whisperSSTis.audio.sd.query_devices', return_value={'default_samplerate': 48000})
@patch('whisperSSTis.audio.sd.InputStream')
@patch('whisperSSTis.audio.AudioStream.get_audio_chunk')
def test_record_audio(mock_get_audio_chunk, _mock_input_stream, _query_devices):
    """Test record_audio function."""
    duration = 1
    device_id = 0
    samplerate = 16000
    
    # Mock audio chunks
    mock_get_audio_chunk.return_value = np.zeros(1024, dtype=np.float32)
    
    audio_data = audio.record_audio(duration, device_id, samplerate)
    
    assert len(audio_data) == duration * samplerate
    assert isinstance(audio_data, np.ndarray)

def test_get_file_info():
    """Test get_file_info function."""
    audio_data = np.zeros(16000, dtype=np.float32)
    sample_rate = 16000
    file_info = audio.get_file_info(audio_data, sample_rate)
    assert "Duration" in file_info
    assert "Channels" in file_info
    assert "Sample Rate" in file_info
    assert "File Size" in file_info

def test_load_audio_file(mocker):
    """Test load_audio_file function."""
    # Mock the uploaded file
    class MockUploadedFile:
        def __init__(self, name, content):
            self.name = name
            self.content = content
        def getvalue(self):
            return self.content

    # Create a mock audio file
    mock_audio_content = b"Mock audio data"
    mock_uploaded_file = MockUploadedFile("test.wav", mock_audio_content)

    # Mock sf.read and other functions
    mocker.patch("whisperSSTis.audio.sf.read", return_value=(np.zeros(16000), 16000))
    mocker.patch("whisperSSTis.audio.os.unlink")

    # Call the function
    audio_data, duration, file_info = audio.load_audio_file(mock_uploaded_file)

    # Assertions
    assert isinstance(audio_data, np.ndarray)
    assert duration == 1.0
    assert isinstance(file_info, dict)

def test_load_m4a_file(mocker):
    """Test load_audio_file function with m4a file."""
    # Mock the uploaded file
    class MockUploadedFile:
        def __init__(self, name, content):
            self.name = name
            self.content = content
        def getvalue(self):
            return self.content

    # Create a mock m4a file
    mock_audio_content = b"Mock m4a audio data"
    mock_uploaded_file = MockUploadedFile("test.m4a", mock_audio_content)

    # Mock AudioSegment and sf.read
    mock_audio_segment = mocker.MagicMock()
    mocker.patch("whisperSSTis.audio.AudioSegment.from_file", return_value=mock_audio_segment)
    mocker.patch("whisperSSTis.audio.sf.read", return_value=(np.zeros(16000), 16000))
    mocker.patch("whisperSSTis.audio.os.unlink")

    # Call the function
    audio_data, duration, file_info = audio.load_audio_file(mock_uploaded_file)

    # Assertions
    assert isinstance(audio_data, np.ndarray)
    assert duration == 1.0
    assert isinstance(file_info, dict)

def test_load_audio_file_exception(mocker):
    """Test load_audio_file function with exception."""
    # Mock the uploaded file
    class MockUploadedFile:
        def __init__(self, name, content):
            self.name = name
            self.content = content
        def getvalue(self):
            return self.content

    # Create a mock audio file
    mock_uploaded_file = MockUploadedFile("test.wav", b"Mock audio data")

    # Mock sf.read to raise an exception
    mocker.patch("whisperSSTis.audio.sf.read", side_effect=Exception("Test exception"))

    # Call the function and assert that it raises an exception
    with pytest.raises(Exception, match="Test exception"):
        audio.load_audio_file(mock_uploaded_file)

@patch('whisperSSTis.audio.sd.query_devices', return_value={'default_samplerate': 48000})
@patch('whisperSSTis.audio.sd.InputStream')
def test_audio_stream_resampling(mock_input_stream, _query_devices, mocker):
    """Test AudioStream resampling functionality."""
    # Mock stream object
    mock_stream = mock_input_stream.return_value
    mock_stream.active = False
    
    # Create AudioStream with different target rate
    stream = audio.AudioStream(device_id=0, samplerate=32000, chunk_size=1024)
    stream.native_samplerate = 48000
    
    # Mock audio callback
    mock_audio_callback = mocker.MagicMock()
    stream._audio_callback = mock_audio_callback
    
    # Mock get_audio_chunk
    mock_chunk = np.zeros(1024, dtype=np.float32)
    stream.get_audio_chunk = mocker.MagicMock(return_value=mock_chunk)
    
    # Start stream
    stream.start_stream()
    
    # Get audio chunk
    chunk = stream.get_audio_chunk()
    
    # Assertions
    assert len(chunk) == 1024  # Check chunk size
    assert stream.target_samplerate == 32000
    assert stream.native_samplerate == 48000
    
    # Stop stream
    stream.stop_stream()
