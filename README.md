# 🎙️ Norðlenski hreimurinn

Real-time Icelandic Speech Recognition powered by Whisper AI

## 🌟 Overview

WhisperSST.is is a 100% local web application that provides real-time Icelandic speech recognition using a fine-tuned version of OpenAI's Whisper model. This tool runs entirely on your machine - no cloud services or internet connection required for processing (only needed for initial model download). Your audio data never leaves your computer, ensuring complete privacy and security.

**Note:** This application is currently in development, so bugs are expected.

## ✨ Features

- 🎤 Record and transcribe audio directly from your microphone
- 📁 Upload and process audio files (WAV, MP3, M4A, FLAC)
- 🔒 100% local processing - no cloud or internet needed
- 🚀 Fast, efficient transcription
- 🔊 Instant audio playback
- 📱 User-friendly interface
- 🇮🇸 Specialized for Icelandic language
- 💻 Runs on your hardware (CPU/GPU)
- 📝 Timestamped transcriptions with chunk-based processing
- 💾 Export to TXT and SRT formats
- 🧠 Optional GPT assistant for post-processing (summarization, translation, cleanup)
- ⚙️ Configurable chunk sizes for optimal processing
- 📊 Real-time processing progress and estimates
- 🎨 Modern, user-friendly interface with light theme

## 🚀 Future Development

- 🎙️ Live transcription feature for real-time speech-to-text conversion
- 📊 Support for more audio formats
- 🧠 Improved accuracy through model fine-tuning
- 📚 Batch processing for multiple files
- 👥 Speaker diarization
- 📄 Export to more formats (DOCX, PDF)
- 🇮🇸 Icelandic translation of the user interface

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended, but CPU works too)
- Microphone access
- Internet connection (only for initial model download)
- ~4GB disk space for models

### Privacy & Security
- 🔒 **Core transcription is 100% local** - your audio never leaves your computer
- 💻 All Whisper transcription happens on your machine
- 🔐 No internet needed after initial model download
- 🧠 **GPT features are optional** - requires API key and sends text (not audio) to OpenAI
- 🎯 You control what data is shared with external services

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio
```

#### macOS
```bash
brew install portaudio
```

#### Windows
The required libraries are typically included with Python packages.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Magnussmari/whisperSSTis.git
cd whisperSSTis
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. **(Optional)** Configure GPT assistant for post-processing:

Create a `.env` file or set environment variables:
```bash
# Required for GPT features
export OPENAI_API_KEY="sk-your-api-key"

# Optional: customize GPT model and endpoint
export GPT_MINI_MODEL="gpt-4o-mini"  # or gpt-3.5-turbo, gpt-4, etc.
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Alternatively, copy `env.example` to `.env` and fill in your values.

**Note:** The GPT assistant is completely optional. The core transcription functionality works without it.

5. Start the application:
```bash
python launcher.py
```

### Development Setup

For developers who want to contribute or modify the application:

1. Set up your development environment:
```bash
# Clone the repository
git clone https://github.com/Magnussmari/whisperSSTis.git
cd whisperSSTis

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. Project Structure:
```
whisperSSTis/
├── app.py                  # Main Streamlit web interface
├── launcher.py             # GUI launcher with process management
├── whisperSSTis/           # Core Python module
│   ├── __init__.py
│   ├── audio.py           # Audio recording, device management, format conversion
│   ├── transcribe.py      # Whisper model integration, chunked processing
│   └── gpt.py             # Optional GPT post-processing helper
├── tests/                  # Unit tests
│   ├── test_audio.py
│   └── test_transcribe.py
├── .streamlit/             # Streamlit configuration
│   └── config.toml
├── requirements.txt        # Python dependencies
├── env.example            # Environment variable template
└── setup_dependencies.*   # System dependency installation scripts
```

3. Running in Development Mode:
```bash
# Run with launcher GUI
python launcher.py

# Run Streamlit directly
streamlit run app.py
```

4. Development Guidelines:
- Follow PEP 8 style guidelines
- Add docstrings for new functions
- Update TODO.md for new features/fixes
- Test changes with different audio inputs

### Running Tests

The project includes comprehensive unit tests for audio and transcription modules:

```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_audio.py
pytest tests/test_transcribe.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=whisperSSTis
```

### Troubleshooting

#### Common Issues

- **Application won't start**: 
  - Run the setup script for your platform
  - Make sure you have extracted all files from the downloaded package
  - Try running as administrator
  - Check your antivirus isn't blocking the application

- **No audio input**: 
  - Run the setup script to install audio dependencies
  - Check your microphone is properly connected
  - Allow microphone access in your system settings
  - Select the correct input device in the application

- **Slow transcription**:
  - A GPU is recommended but not required
  - First launch may be slow while loading the model
  - Try adjusting chunk size for better performance
  - Models are cached locally for faster subsequent runs

- **PortAudio Error**: 
  - Run `setup_dependencies.sh` (macOS/Linux) or `setup_dependencies.bat` (Windows)
  - Windows: Install Visual C++ Redistributable if prompted
  - Linux: Run `sudo apt-get install portaudio19-dev python3-pyaudio`
  - macOS: Run `brew install portaudio`

- **Missing Dependencies**:
  - Run the setup script for your platform
  - Check the error message for specific missing packages
  - For Windows, ensure Visual C++ Redistributable is installed
  - For Linux, install required system packages using your package manager

For more help, check the [issues page](https://github.com/Magnussmari/whisperSSTis/issues) or create a new issue.

## 💻 Technical Details

### Core Stack
- **Frontend**: Streamlit with custom CSS (modern light theme)
- **Speech Recognition**: Fine-tuned Whisper model (`carlosdanielhernandezmena/whisper-large-icelandic-10k-steps-1000h`)
- **Audio Processing**: sounddevice, soundfile, PortAudio
- **ML Framework**: PyTorch, Hugging Face Transformers
- **Optional AI**: OpenAI API for post-processing

### Processing Details
- **Sample Rate**: 16kHz (Whisper-optimized)
- **Chunk Processing**: Configurable 10-60 second segments
- **Device Support**: Automatic GPU (CUDA) detection with CPU fallback
- **Audio Formats**: WAV, MP3, M4A, FLAC (via FFmpeg)
- **Max Upload**: 1000 MB files supported
- **Privacy**: Core transcription is 100% local; GPT features are optional and require API key

### GPT Assistant Features
When configured with an OpenAI API key, the application offers:
- **Summarization**: Generate concise summaries in Icelandic or English
- **Translation**: Translate transcripts between languages
- **Cleanup**: Improve wording, fix grammar, or reformat text
- **Custom Instructions**: Flexible prompts for any post-processing task
- **Configurable**: Adjust temperature and token limits per request

## 👥 Credits

### Developer
- **Magnus Smari Smarason**

### Model Credits
- **Original Whisper Model**: [OpenAI](https://github.com/openai/whisper)
- **Icelandic Fine-tuned Model**: [Carlos Daniel Hernandez Mena](https://huggingface.co/carlosdanielhernandezmena/whisper-large-icelandic-10k-steps-1000h)

### Technologies
- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Magnussmari/whisperSSTis/issues).

## 🔒 Security & Privacy

### Data Privacy
- **Audio Processing**: 100% local - audio never transmitted over network
- **Whisper Model**: Downloaded once from Hugging Face, then cached locally
- **GPT Integration** (optional): Only sends transcribed text (not audio) to OpenAI API
- **File Uploads**: Processed locally, stored temporarily, automatically cleaned up

### Security Considerations

#### External Dependencies
- **Hugging Face Model**: Fine-tuned model from reputable source (Carlos Daniel Hernandez Mena)
- **FFmpeg**: Required for format conversion; keep updated for security patches
- **OpenAI API** (optional): Requires API key; only sends text data when explicitly requested

#### Code Security
- **HTML Rendering**: `unsafe_allow_html=True` used only for static CSS styling, not user input
- **Input Validation**: File type and size validation for uploads (max 1000 MB)
- **Error Handling**: Proper cleanup of temporary files in all code paths
- **API Keys**: Environment variable-based configuration (never hardcoded)

### Security Best Practices
1. **Keep Dependencies Updated**: Regularly update `transformers`, `streamlit`, `ffmpeg-python`, and other packages
2. **API Key Management**: Store `OPENAI_API_KEY` in `.env` file (not version controlled)
3. **Offline Mode**: Core functionality works without internet after initial model download
4. **Model Integrity**: Hugging Face models include checksums for verification
5. **Network Isolation**: Consider running offline if handling sensitive audio

### Recommendations
- Monitor dependencies for CVEs using tools like `pip-audit` or Dependabot
- Keep FFmpeg system package updated via your OS package manager
- Review `.gitignore` to ensure `.env` and sensitive files are never committed
- Use the application offline when processing confidential audio
<p align="center">
Developed with ❤️ for the Icelandic language community
</p>
