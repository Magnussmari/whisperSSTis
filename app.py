import math
import os

import soundfile as sf
import streamlit as st
import torch

from whisperSSTis import audio, gpt, transcribe

MODEL_LABEL_TO_KEY = {cfg["label"]: key for key, cfg in transcribe.MODEL_CONFIGS.items()}

# Set page config and theme
st.set_page_config(
    page_title="Norðlenski hreimurinn - Icelandic Speech Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force light mode
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-color: #ffffff;
        }
        .stApp {
            background-color: #ffffff;
        }
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        [data-testid="stMarkdownContainer"] {
            color: #000000;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'processor' not in st.session_state:
    st.session_state['processor'] = None
if 'audio_data' not in st.session_state:
    st.session_state['audio_data'] = None
if 'last_transcript_text' not in st.session_state:
    st.session_state['last_transcript_text'] = ""
if 'gpt_record_response' not in st.session_state:
    st.session_state['gpt_record_response'] = ""
if 'gpt_upload_response' not in st.session_state:
    st.session_state['gpt_upload_response'] = ""
if 'selected_model_key' not in st.session_state:
    st.session_state['selected_model_key'] = transcribe.DEFAULT_MODEL_KEY
if 'language_token' not in st.session_state:
    st.session_state['language_token'] = transcribe.get_model_config()['language_token']

# Sidebar
with st.sidebar:
    st.image("assets/websitelogo.png", width=200)
    st.title("Norðlenski hreimurinn")
    st.caption("🧪 Alpha Version")
    st.markdown("---")
    st.subheader("🧪 About")
    st.markdown("Norðlenski hreimurinn provides real-time Icelandic speech recognition using OpenAI's Whisper model, fine-tuned for Icelandic.")
    st.markdown("Developed by [**Magnus Smari Smarason**](https://www.smarason.is)")
    st.markdown("---")
    st.subheader("🏆 Credits")
    st.markdown("""
    - **Original Whisper Model**: [OpenAI](https://github.com/openai/whisper)
    - **Icelandic Fine-tuned Model**: [Carlos Daniel Hernandez Mena](https://huggingface.co/carlosdanielhernandezmena/whisper-large-icelandic-10k-steps-1000h)
    - Built with [Streamlit](https://streamlit.io/), [PyTorch](https://pytorch.org/), and [Hugging Face](https://huggingface.co/)
    """)
    st.markdown("---")
    st.subheader("🔧 System Info")
    device_status = "🖥️ GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
    st.info(f"Using: {device_status}")
    if torch.cuda.is_available():
        st.success(f"GPU: {torch.cuda.get_device_name(0)}")
    st.markdown("---")
    st.subheader("🗣️ Language & Model")
    model_labels = list(MODEL_LABEL_TO_KEY.keys())
    active_label = transcribe.get_model_config(st.session_state['selected_model_key'])["label"]
    selected_label = st.selectbox(
        "Choose speech model",
        options=model_labels,
        index=model_labels.index(active_label),
        help="First run downloads the model weights locally. Choosing a new model reloads the session.",
    )
    selected_key = MODEL_LABEL_TO_KEY[selected_label]
    if selected_key != st.session_state['selected_model_key']:
        st.session_state['selected_model_key'] = selected_key
        st.session_state['model'] = None
        st.session_state['processor'] = None
    chosen_config = transcribe.get_model_config(st.session_state['selected_model_key'])
    st.caption(f"{chosen_config['label']} · {chosen_config['model_id']}")

st.title("Multilingual Speech Recognition 🎙️")
st.caption("Switch between Icelandic and English Whisper models for local transcription")


def render_gpt_assistant(transcript_text: str, response_key: str):
    """Render the GPT mini helper panel when transcript text exists."""

    if not transcript_text:
        return

    with st.expander("💡 GPT-5 mini assistant", expanded=False):
        st.caption("Use GPT-5 mini to summarize, clean up wording, or translate your transcript. Requires OPENAI_API_KEY.")
        default_prompt = "Skrifaðu stutta samantekt á íslensku og bættu við helstu lykilatriðum."
        prompt_value = st.session_state.get(f"prompt_{response_key}", default_prompt)
        prompt = st.text_area(
            "Instruction",
            value=prompt_value,
            help="Tell GPT-5 mini what to do with the transcript (e.g. summarize, translate, bullet points).",
            key=f"prompt_input_{response_key}"
        )

        col_a, col_b = st.columns([3, 1])
        with col_a:
            temperature = st.slider("Creativity", min_value=0.0, max_value=1.0, value=0.3, step=0.05, key=f"temp_{response_key}")
        with col_b:
            max_tokens = st.number_input("Max tokens", min_value=100, max_value=1200, value=400, step=50, key=f"tokens_{response_key}")

        if st.button("Ask GPT-5 mini", key=f"ask_gpt_{response_key}", use_container_width=True):
            if not os.getenv("OPENAI_API_KEY"):
                st.warning("Set OPENAI_API_KEY to enable GPT mini features.")
            else:
                try:
                    with st.spinner("🧠 Contacting GPT-5 mini..."):
                        response = gpt.run_on_transcript(
                            transcript_text,
                            instruction=prompt,
                            config=gpt.GPTConfig(temperature=temperature, max_tokens=int(max_tokens)),
                        )
                    st.session_state[response_key] = response
                    st.session_state[f"prompt_{response_key}"] = prompt
                except Exception as e:
                    st.error(f"GPT mini error: {e}")

        if st.session_state.get(response_key):
            st.markdown("###### GPT-5 mini response")
            st.write(st.session_state[response_key])

# Load the model if not loaded
if st.session_state['model'] is None or st.session_state['processor'] is None:
    selected_key = st.session_state['selected_model_key']
    config = transcribe.get_model_config(selected_key)
    with st.spinner(f"🤖 Loading {config['label']}... This might take a minute..."):
        model, processor = transcribe.load_model(selected_key)
        st.session_state['model'] = model
        st.session_state['processor'] = processor
        st.session_state['language_token'] = config['language_token']
    st.success(f"✅ {config['label']} ready!")

# Create tabs
tabs = ["🎤 Record Audio", "📁 Upload Audio"]
current_tab = st.radio("Select Mode", tabs, horizontal=True, label_visibility="collapsed")

# Set the active tab based on selection
active_tab = tabs.index(current_tab)

# Handle tab content based on active tab
if active_tab == 0:
    st.markdown("""
    <div class="upload-header">
    <h3>Record Your Voice</h3>
    <p>Speak in Icelandic and get instant transcription.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3,2])
    
    with col1:
        with st.container():
            st.markdown("##### Device Settings")
            input_devices = audio.get_audio_devices()
            if not input_devices:
                st.error("❌ No input devices found!")
            else:
                device_options = list(input_devices.keys())
                device_choice = st.selectbox("🎙️ Select Input Device", options=device_options)
                selected_device_id = input_devices[device_choice]
                rec_duration = st.slider("⏱️ Recording Duration (seconds)", min_value=1, max_value=30, value=5, help="Choose how long you want to record")
                if st.button("🎤 Start Recording", use_container_width=True):
                    try:
                        with st.spinner("🎤 Recording in progress..."):
                            audio_data = audio.record_audio(rec_duration, selected_device_id)
                        st.session_state['audio_data'] = audio_data
                        
                        temp_path = "temp_recording.wav"
                        sf.write(temp_path, audio_data, 16000)
                        st.audio(temp_path)
                        
                        with st.spinner("🤖 Processing your speech..."):
                            transcription = transcribe.transcribe_audio(
                                audio_data,
                                st.session_state['model'],
                                st.session_state['processor'],
                                language_token=st.session_state['language_token'],
                            )
                        st.session_state['last_transcript_text'] = transcription
                        st.session_state['gpt_record_response'] = ""

                        st.markdown("##### 📝 Transcription")
                        st.markdown(f'<div class="file-info">{transcription}</div>', unsafe_allow_html=True)
                        os.remove(temp_path)
                        render_gpt_assistant(transcription, "gpt_record_response")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.markdown("##### 📋 Instructions")
        st.info("""
        1. Select your microphone from the dropdown.
        2. Set your preferred recording duration.
        3. Click "Start Recording" and speak in Icelandic.
        4. Wait for the transcription to appear.
        
        **Note:** Ensure your microphone is properly connected and permitted.
        """)

elif active_tab == 1:
    st.markdown("""
    <div class="upload-header">
    <h3>Upload Audio File</h3>
    <p>Process longer recordings like podcasts or interviews.</p>
    </div>
    """, unsafe_allow_html=True)
    
    upload_col1, upload_col2 = st.columns([3,2])
    
    with upload_col1:
        st.markdown("##### 📁 Upload Your File")
        # Add key to file uploader to maintain state
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'flac'],
            help="Supported formats: WAV, MP3, M4A, FLAC",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            # Store file info in session state
            if 'uploaded_file_info' not in st.session_state:
                st.session_state['uploaded_file_info'] = {
                    'name': uploaded_file.name,
                    'type': uploaded_file.type,
                    'size': uploaded_file.size
                }
            
            st.audio(uploaded_file)
            
            try:
                with st.spinner("🤖 Analyzing file..."):
                    audio_data, duration, file_info = audio.load_audio_file(uploaded_file)
                if audio_data is not None and file_info is not None:
                    # Store processing results in session state
                    if 'processing_results' not in st.session_state:
                        st.session_state['processing_results'] = {
                            'audio_data': audio_data,
                            'duration': duration,
                            'file_info': file_info
                        }
                    
                    st.markdown("##### 📊 File Details")
                    detail_cols = st.columns(len(file_info))
                    for col, (key, value) in zip(detail_cols, file_info.items()):
                        col.metric(key, value)
                    
                    st.markdown("##### ⚙️ Processing Options")
                    chunk_size = st.slider("Segment Length", min_value=10, max_value=60, value=30, 
                                         help="Length of each transcribed segment. Shorter segments may improve accuracy but take longer to process.")
                    
                    # Calculate number of segments
                    num_segments = math.ceil(duration / chunk_size)
                    est_processing_time = num_segments * 2  # Rough estimate in seconds
                    
                    st.info(f"""
                    📊 **Estimated Processing:**
                    - Segments: {num_segments}
                    - Processing time: ~{transcribe.format_timestamp(est_processing_time)}
                    - Audio length: {transcribe.format_timestamp(duration)}
                    """)
                    
                    if st.button("🚀 Start Processing", use_container_width=True):
                        with st.spinner("🤖 Processing your audio..."):
                            transcriptions = transcribe.transcribe_long_audio(
                                audio_data,
                                st.session_state['model'],
                                st.session_state['processor'],
                                duration,
                                chunk_size,
                                language_token=st.session_state['language_token'],
                            )
                        st.session_state['last_transcript_text'] = "\n".join(transcriptions)
                        st.session_state['gpt_upload_response'] = ""
                        
                        st.markdown("##### 💾 Download Options")
                        dl_col1, dl_col2 = st.columns(2)
                        with dl_col1:
                            full_text = "\n\n".join(transcriptions)
                            st.download_button("📄 Download TXT", full_text, file_name=f"{uploaded_file.name}_transcript.txt", mime="text/plain", use_container_width=True)
                        with dl_col2:
                            srt_content = transcribe.create_srt(transcriptions)
                            st.download_button("🎬 Download SRT", srt_content, file_name=f"{uploaded_file.name}_transcript.srt", mime="text/plain", use_container_width=True)
                        
                        st.markdown("##### 📝 Transcription")
                        for trans in transcriptions:
                            st.markdown(f'<div class="file-info">{trans}</div>', unsafe_allow_html=True)
                        render_gpt_assistant("\n".join(transcriptions), "gpt_upload_response")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with upload_col2:
        st.markdown("##### ℹ️ About File Upload")
        st.info("""
        **Supported Features:**
        - Long audio files (podcasts, interviews)
        - Multiple audio formats
        - Timestamped transcription
        - Download as TXT or SRT
        
        **Tips:**
        - For best results, use clear audio.
        - Larger files will take longer to process.
        - You can adjust segment length for better accuracy.
        """)

# Add custom styles
st.markdown("""
    <style>
    .live-transcript {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
        min-height: 200px;
        max-height: 400px;
        overflow-y: auto;
        white-space: pre-wrap;
        font-size: 1.1em;
        line-height: 1.6;
    }
    .main > div {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
    }
    .stButton > button {
        width: 100%;
    }
    .upload-header {
        margin-bottom: 2rem;
    }
    .file-info {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .css-1dp5vir {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .css-1dp5vir:hover {
        box-shadow: 0 3px 6px rgba(0,0,0,0.16);
        transition: all 0.2s ease;
    }
    /* Improve tab transitions */
    .stTabs [data-baseweb="tab-list"] button {
        transition: all 0.3s ease-in-out;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        border-bottom-color: rgb(255, 75, 75);
    }
    .stTabs [data-baseweb="tab-panel"] {
        transition: opacity 0.3s ease-in-out;
    }
    /* Style radio buttons to look like tabs */
    div.row-widget.stRadio > div {
        flex-direction: row;
        justify-content: center;
        gap: 2rem;
    }
    
    div.row-widget.stRadio > div[role="radiogroup"] > label {
        padding: 0.5rem 1rem;
        background: none;
        border-radius: 4px;
        cursor: pointer;
        margin: 0;
        transition: all 0.3s ease;
    }
    
    div.row-widget.stRadio > div[role="radiogroup"] > label:hover {
        background: rgba(255, 75, 75, 0.1);
    }
    
    div.row-widget.stRadio > div[role="radiogroup"] label[data-baseweb="radio"] {
        border-color: transparent;
    }
    
    div.row-widget.stRadio > div[role="radiogroup"] > label > div {
        background-color: transparent !important;
        border-color: transparent !important;
    }
    
    /* Hide the radio button circle */
    div.row-widget.stRadio > div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
        display: none;
    }
    
    /* Add bottom border for selected tab */
    div.row-widget.stRadio > div[role="radiogroup"] label[data-baseweb="radio"][aria-checked="true"] {
        border-bottom: 2px solid rgb(255, 75, 75);
    }
    </style>
""", unsafe_allow_html=True)
