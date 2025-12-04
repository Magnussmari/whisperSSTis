# Project Content Template for magnussmari.com

## Purpose
Simple template for preparing project content that matches the Sanity schema and creates consistent project pages.

## Project Information (Matches Sanity Schema)

### Basic Fields
- **Title (EN)**: WhisperSST.is - Icelandic Speech Recognition
- **Title (IS)**: WhisperSST.is - Norðlenski hreimurinn
- **Slug**: whisper-sst-is
- **Category**: AI
- **Duration**: 3 months
- **Featured**: Yes
- **Published Date**: 2024-11-09

### URLs
- **Demo URL**: N/A (Local application)
- **GitHub URL**: https://github.com/Magnussmari/whisperSSTis

### Content Fields
- **Description (EN)**: A privacy-focused speech recognition tool for Icelandic that runs entirely on your local machine. Built with a fine-tuned Whisper AI model, it provides accurate transcriptions without sending your audio data to the cloud. Perfect for transcribing podcasts, interviews, and meetings while maintaining complete data privacy.
- **Description (IS)**: Einkatölvubundið talgreiningartól fyrir íslensku sem keyrir alfarið á þinni tölvu. Byggt með sérþjálfuðu Whisper AI líkani, veitir það nákvæmar uppskriftir án þess að senda hljóðgögnin þín í skýið. Tilvalið til að skrifa upp hlaðvörp, viðtöl og fundi með fullri gagnavernd.
- **Excerpt (EN)**: Local Icelandic speech recognition with Whisper AI. Privacy-first design for podcasts and interviews.
- **Excerpt (IS)**: Staðbundin íslensk talgreining með Whisper AI. Persónuvernd í fyrirrúmi fyrir hlaðvörp og viðtöl.

### Technologies
List the main technologies used:
- Python
- Streamlit
- PyTorch
- Whisper AI (Hugging Face)
- Transformers
- FFmpeg

### Main Content
Write the full project story using simple markdown:

```markdown
## Overview
WhisperSST.is brings high-quality speech recognition to the Icelandic language community with a focus on privacy and local processing. Using a specially fine-tuned version of OpenAI's Whisper model trained on 1000+ hours of Icelandic speech, it achieves impressive accuracy while running entirely on consumer hardware.

The project was born from my need to transcribe Icelandic podcast episodes efficiently without compromising data privacy. Traditional cloud-based solutions require uploading sensitive audio, which isn't ideal for private conversations or confidential content. This tool solves that by keeping everything local.

## Key Features
- **100% Local Processing**: Audio never leaves your computer
- **Real-time Recording**: Record and transcribe directly from your microphone
- **File Upload Support**: Process WAV, MP3, M4A, and FLAC files
- **Export Options**: Save transcripts as TXT or SRT subtitle files
- **GPU Acceleration**: Utilizes CUDA when available for faster processing
- **Timestamped Output**: Perfect for creating subtitles or finding specific moments

## Technical Implementation
The application uses Streamlit for a clean web interface that runs locally. The core transcription engine leverages a Whisper Large model fine-tuned specifically for Icelandic by Carlos Daniel Hernandez Mena. Audio processing handles resampling to 16kHz (Whisper's required rate) and chunks long recordings into 30-second segments for memory efficiency.

A custom Tkinter launcher manages the Streamlit process, handling port allocation and browser integration. The system automatically detects available audio devices and validates their compatibility with the required sample rates.

## Results/Impact
The tool now serves as my primary transcription solution for podcast production, significantly reducing the time needed to create show notes and searchable content. Running on a consumer laptop with an RTX 3060, it processes audio at approximately 2-3x real-time speed, making hour-long episodes manageable.

The project demonstrates that privacy-respecting AI tools can be both practical and performant on consumer hardware, opening possibilities for similar local-first applications in other domains.
```