# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development & Testing
```bash
# Run application with GUI launcher
python launcher.py

# Run Streamlit app directly
streamlit run app.py

# Run tests
pytest

# Install dependencies
pip install -r requirements.txt

# Run specific test file
pytest tests/test_audio.py
pytest tests/test_transcribe.py
```

### System Dependencies
```bash
# macOS
brew install portaudio

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio

# Windows
# Run setup_dependencies.bat
```

## Architecture

### Core Components

**whisperSSTis/** - Main module for audio processing and transcription
- `audio.py`: Audio recording/processing, device management, format conversion
- `transcribe.py`: Whisper model integration, transcription with timestamps

**app.py** - Streamlit web interface
- Recording and file upload functionality
- Real-time transcription display
- Export to TXT/SRT formats
- Device selection UI

**launcher.py** - Tkinter GUI launcher
- Manages Streamlit process lifecycle
- Port management and browser integration
- Process monitoring

### Model & Processing

- Uses `carlosdanielhernandezmena/whisper-large-icelandic-10k-steps-1000h` model
- 16kHz sample rate for Whisper compatibility
- Supports resampling from native device rates
- GPU acceleration when available (falls back to CPU)

### Audio Format Support
- Direct: WAV, FLAC
- Via FFmpeg: MP3, M4A
- Handles mono/stereo conversion automatically

### Key Technical Details

- **Privacy**: 100% local processing, no cloud dependencies after model download
- **State Management**: Uses Streamlit session state for audio data and transcriptions
- **Chunk Processing**: 30-second chunks for efficient memory usage
- **Timestamp Generation**: Word-level timing for SRT export