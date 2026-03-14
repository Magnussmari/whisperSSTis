# Agents.md - WhisperSST.is Project

This document provides guidance for AI agents working with the WhisperSST.is project, an Icelandic speech recognition application powered by Whisper AI.

## Project Overview

**WhisperSST.is** is a 100% local web application that provides real-time Icelandic speech recognition using a fine-tuned version of OpenAI's Whisper model. The application runs entirely on the user's machine with no cloud dependencies after the initial model download.

### Key Features
- 🎤 Real-time audio recording and transcription
- 📁 Audio file upload and processing (WAV, MP3, M4A, FLAC)
- 🔒 Complete privacy - all processing happens locally
- 🇮🇸 Specialized for Icelandic language
- 📝 Timestamped transcriptions with SRT export
- 💻 GPU/CPU acceleration support

## Architecture

### Core Components

```
whisperSSTis/
├── app.py                 # Main Streamlit web interface
├── launcher.py           # Tkinter GUI launcher
├── whisperSSTis/         # Core module
│   ├── __init__.py
│   ├── audio.py          # Audio recording/processing
│   └── transcribe.py     # Whisper model integration
├── requirements.txt      # Python dependencies
└── assets/              # Static assets (logos, etc.)
```

### Technology Stack
- **Frontend**: Streamlit (local web interface)
- **ML Framework**: PyTorch, Hugging Face Transformers
- **Audio Processing**: PortAudio, sounddevice, soundfile
- **Model**: `carlosdanielhernandezmena/whisper-large-icelandic-10k-steps-1000h`

## Development Commands

### Essential Commands
```bash
# Run application with GUI launcher
python launcher.py

# Run Streamlit app directly
streamlit run app.py

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

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

## Code Structure & Key Functions

### Audio Module (`whisperSSTis/audio.py`)
- `get_audio_devices()` - Lists available input devices
- `record_audio(duration, device_id, samplerate)` - Records audio from microphone
- `load_audio_file(uploaded_file, target_sr)` - Processes uploaded audio files
- `get_file_info(audio_data, sample_rate)` - Extracts audio metadata

### Transcription Module (`whisperSSTis/transcribe.py`)
- `load_model()` - Loads the Whisper model and processor
- `transcribe_audio(audio_data, model, processor)` - Transcribes audio chunks
- `transcribe_long_audio(audio_data, model, processor, duration, chunk_size)` - Handles long audio files
- `create_srt(transcriptions)` - Generates SRT subtitle format

### Main Application (`app.py`)
- Streamlit web interface with two main tabs:
  - **Record Audio**: Real-time microphone recording
  - **Upload Audio**: File upload and processing
- Session state management for model and audio data
- Export functionality (TXT and SRT formats)

### Launcher (`launcher.py`)
- Tkinter GUI for launching the Streamlit app
- Port management and process monitoring
- Browser integration

## Key Technical Details

### Audio Processing
- **Sample Rate**: 16kHz (required by Whisper)
- **Channels**: Mono (automatically converted from stereo)
- **Format Support**: WAV, MP3, M4A, FLAC (via FFmpeg)
- **Chunk Processing**: 30-second segments for long audio files

### Model Configuration
- **Model**: `carlosdanielhernandezmena/whisper-large-icelandic-10k-steps-1000h`
- **Language**: Icelandic (`<|is|>` language tag)
- **Device**: CUDA if available, falls back to CPU
- **Privacy**: 100% local processing

### State Management
- Uses Streamlit session state for:
  - Model and processor instances
  - Audio data storage
  - Processing results
  - File upload information

## Common Tasks for AI Agents

### 1. Adding New Features
- **Audio Format Support**: Extend `load_audio_file()` in `audio.py`
- **Export Formats**: Add new export functions in `transcribe.py`
- **UI Components**: Modify `app.py` Streamlit interface
- **Model Integration**: Update `transcribe.py` for different models

### 2. Debugging Issues
- **Audio Recording Problems**: Check `get_audio_devices()` and PortAudio installation
- **Model Loading Issues**: Verify Hugging Face model availability and CUDA setup
- **File Processing Errors**: Check FFmpeg installation and file format support
- **Memory Issues**: Monitor chunk size in `transcribe_long_audio()`

### 3. Performance Optimization
- **GPU Utilization**: Ensure CUDA is properly configured
- **Memory Management**: Adjust chunk sizes for large files
- **Model Caching**: Leverage Hugging Face model caching
- **Audio Resampling**: Optimize `signal.resample()` calls

### 4. Testing
- **Unit Tests**: Test individual functions in `whisperSSTis/` modules
- **Integration Tests**: Test full workflow from recording to transcription
- **Audio Tests**: Test with various audio formats and qualities
- **Model Tests**: Verify transcription accuracy with known audio samples

## Security Considerations

### Current Security Measures
- ✅ 100% local processing (no cloud dependencies)
- ✅ Temporary file cleanup
- ✅ Input validation for audio files
- ✅ Safe HTML rendering (static content only)

### Areas for Improvement
- 🔍 Model integrity verification
- 🔍 Dependency vulnerability monitoring
- 🔍 Input sanitization for user data
- 🔍 Error message sanitization

## Development Guidelines

### Code Style
- Follow PEP 8 guidelines
- Add comprehensive docstrings
- Use type hints where appropriate
- Handle exceptions gracefully

### Testing Strategy
- Unit tests for core functions
- Integration tests for workflows
- Audio quality tests with various inputs
- Performance tests with different file sizes

### Error Handling
- Log errors with appropriate detail
- Provide user-friendly error messages
- Clean up temporary files in all scenarios
- Graceful fallbacks for missing dependencies

## Common Issues & Solutions

### PortAudio Errors
```bash
# macOS
brew install portaudio

# Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio
```

### Model Loading Issues
- Check internet connection for initial download
- Verify CUDA installation for GPU acceleration
- Monitor disk space (models are ~4GB)

### Audio Device Problems
- Check microphone permissions
- Verify device selection in UI
- Test with different sample rates

### Memory Issues
- Reduce chunk size for large files
- Monitor GPU memory usage
- Consider CPU fallback for very large files

## Future Development Areas

### Planned Features
- 🎙️ Live transcription
- 📊 More audio formats
- 🧠 Model fine-tuning
- 📚 Batch processing
- 👥 Speaker diarization
- ⏱️ Word-level timestamps

### Technical Improvements
- Better error handling
- Performance optimization
- UI/UX enhancements
- Mobile responsiveness
- Offline model verification

## Resources

### Documentation
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [PortAudio Documentation](http://www.portaudio.com/docs.html)

### Model Information
- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [Icelandic Model](https://huggingface.co/carlosdanielhernandezmena/whisper-large-icelandic-10k-steps-1000h)

### Community
- [GitHub Issues](https://github.com/Magnussmari/whisperSSTis/issues)
- [Developer Contact](https://www.smarason.is)

---

*This document is maintained alongside the project codebase. Update it when making significant changes to the architecture or functionality.*