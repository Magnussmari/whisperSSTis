"""WhisperSST.is — Local speech recognition for Icelandic and English."""

import math
import os
import tempfile

import numpy as np
import soundfile as sf
import streamlit as st
import torch

from whisperSSTis import audio, gpt, transcribe

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Norðlenski hreimurinn",
    page_icon="assets/websitelogo.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_LABEL_TO_KEY = {cfg["label"]: key for key, cfg in transcribe.MODEL_CONFIGS.items()}
BRAND = "#2563eb"  # Blue-600

# ---------------------------------------------------------------------------
# Design system — single CSS block
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
    /* Base */
    .stApp {{ background: #f8fafc; }}
    section[data-testid="stSidebar"] {{ background: #ffffff; border-right: 1px solid #e2e8f0; }}

    /* Header bar */
    .app-header {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 1rem 0 0.75rem; border-bottom: 1px solid #e2e8f0; margin-bottom: 1.5rem;
    }}
    .app-header h1 {{ margin: 0; font-size: 1.6rem; font-weight: 700; color: #0f172a; }}
    .app-header .subtitle {{ color: #64748b; font-size: 0.85rem; margin-top: 2px; }}

    /* Status badge */
    .badge {{
        display: inline-block; padding: 3px 10px; border-radius: 9999px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
    }}
    .badge-ready {{ background: #dcfce7; color: #166534; }}
    .badge-loading {{ background: #fef9c3; color: #854d0e; }}
    .badge-local {{ background: #eff6ff; color: #1e40af; }}

    /* Cards */
    .card {{
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 1.5rem; margin-bottom: 1rem;
    }}
    .card-title {{ font-size: 1rem; font-weight: 600; color: #0f172a; margin-bottom: 0.75rem; }}

    /* Transcript block */
    .transcript-block {{
        background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid {BRAND};
        border-radius: 8px; padding: 1.25rem; margin: 0.5rem 0; line-height: 1.7;
        font-size: 0.95rem; color: #1e293b;
    }}
    .transcript-placeholder {{
        background: #f1f5f9; border: 2px dashed #cbd5e1; border-radius: 8px;
        padding: 3rem 1.5rem; text-align: center; color: #94a3b8;
    }}

    /* Waveform container */
    .waveform-label {{ font-size: 0.78rem; color: #64748b; font-weight: 500; margin-bottom: 0.25rem; }}

    /* File details pills */
    .file-pill {{
        display: inline-block; background: #f1f5f9; border-radius: 6px;
        padding: 4px 10px; font-size: 0.78rem; color: #475569; margin-right: 6px;
    }}

    /* Hide Streamlit extras */
    #MainMenu {{ visibility: hidden; }}
    header {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}

    /* Button overrides */
    .stButton > button {{
        border-radius: 8px; font-weight: 500; transition: all 0.15s ease;
    }}
    div[data-testid="stDownloadButton"] > button {{
        border-radius: 8px; font-weight: 500;
    }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_defaults = {
    "model": None,
    "processor": None,
    "audio_data": None,
    "last_transcript_text": "",
    "gpt_record_response": "",
    "gpt_upload_response": "",
    "selected_model_key": transcribe.DEFAULT_MODEL_KEY,
    "language_token": transcribe.get_model_config()["language_token"],
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------------------------
# Sidebar — compact: model selector + system info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("assets/websitelogo.png", width=160)
    st.markdown("#### Stillingar / Settings")

    model_labels = list(MODEL_LABEL_TO_KEY.keys())
    active_label = transcribe.get_model_config(st.session_state["selected_model_key"])["label"]
    selected_label = st.selectbox(
        "Tungumál / Language model",
        options=model_labels,
        index=model_labels.index(active_label),
        help="Fyrsta keyrsla sækir módel sjálfkrafa. / First run downloads model weights.",
    )
    selected_key = MODEL_LABEL_TO_KEY[selected_label]
    if selected_key != st.session_state["selected_model_key"]:
        st.session_state["selected_model_key"] = selected_key
        st.session_state["model"] = None
        st.session_state["processor"] = None

    device_label = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.caption(f"Keyrir á: {device_label}")

    st.divider()

    with st.expander("Um / About"):
        st.markdown(
            "**Norðlenski hreimurinn** veitir staðbundna talgreiningu á íslensku "
            "og ensku. Öll vinnsla fer fram á tölvunni þinni — ekkert hljóð fer í skýið."
        )
        st.markdown("Höfundur: [Magnus Smari Smarason](https://smarason.is)")
        st.caption(
            "Whisper módel: [OpenAI](https://github.com/openai/whisper) · "
            "[Íslenskt módel](https://huggingface.co/carlosdanielhernandezmena/whisper-large-icelandic-10k-steps-1000h)"
        )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
model_ready = st.session_state["model"] is not None
badge_cls = "badge-ready" if model_ready else "badge-loading"
badge_txt = "Tilbúið / Ready" if model_ready else "Hleð módeli…"
chosen_config = transcribe.get_model_config(st.session_state["selected_model_key"])

st.markdown(f"""
<div class="app-header">
    <div>
        <h1>Norðlenski hreimurinn</h1>
        <div class="subtitle">{chosen_config['label']} &middot; Staðbundin talgreining / Local speech recognition</div>
    </div>
    <div>
        <span class="badge badge-local">100% staðbundið</span>
        <span class="badge {badge_cls}">{badge_txt}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load model (blocking)
# ---------------------------------------------------------------------------
if st.session_state["model"] is None or st.session_state["processor"] is None:
    config = transcribe.get_model_config(st.session_state["selected_model_key"])
    with st.spinner(f"Hleð {config['label']}… Þetta getur tekið smá stund."):
        model, processor = transcribe.load_model(st.session_state["selected_model_key"])
        st.session_state["model"] = model
        st.session_state["processor"] = processor
        st.session_state["language_token"] = config["language_token"]
    st.rerun()


# ---------------------------------------------------------------------------
# GPT assistant helper
# ---------------------------------------------------------------------------
def render_gpt_assistant(transcript_text: str, response_key: str):
    """Optional GPT post-processing panel."""
    if not transcript_text:
        return

    with st.expander("AI aðstoð / AI Assistant (GPT)", expanded=False):
        st.caption("Notaðu GPT til að draga saman, þýða eða hreinsa texta. Krefst OPENAI_API_KEY.")
        default_prompt = "Skrifaðu stutta samantekt á íslensku og bættu við helstu lykilatriðum."
        prompt = st.text_area(
            "Fyrirmæli / Instruction",
            value=st.session_state.get(f"prompt_{response_key}", default_prompt),
            key=f"prompt_input_{response_key}",
        )

        col_a, col_b = st.columns([3, 1])
        with col_a:
            temperature = st.slider("Sköpunarkraftur", 0.0, 1.0, 0.3, 0.05, key=f"temp_{response_key}")
        with col_b:
            max_tokens = st.number_input("Hámark orða", 100, 1200, 400, 50, key=f"tokens_{response_key}")

        if st.button("Senda / Send", key=f"ask_gpt_{response_key}", use_container_width=True):
            if not os.getenv("OPENAI_API_KEY"):
                st.warning("Settu OPENAI_API_KEY til að virkja AI aðstoð.")
            else:
                try:
                    with st.spinner("Sendi til GPT…"):
                        response = gpt.run_on_transcript(
                            transcript_text,
                            instruction=prompt,
                            config=gpt.GPTConfig(temperature=temperature, max_tokens=int(max_tokens)),
                        )
                    st.session_state[response_key] = response
                    st.session_state[f"prompt_{response_key}"] = prompt
                except Exception as e:
                    st.error(f"Villa: {e}")

        if st.session_state.get(response_key):
            st.markdown("**Svar frá GPT:**")
            st.write(st.session_state[response_key])


# ---------------------------------------------------------------------------
# Main content — two tabs
# ---------------------------------------------------------------------------
tab_record, tab_upload = st.tabs(["Upptaka / Record", "Hlaða upp / Upload"])

# ========== TAB: RECORD ====================================================
with tab_record:
    controls_col, results_col = st.columns([2, 3], gap="large")

    with controls_col:
        st.markdown('<div class="card"><div class="card-title">Upptaka / Recording</div>', unsafe_allow_html=True)

        input_devices = audio.get_audio_devices()
        if not input_devices:
            st.error("Ekkert hljóðnema fannst. Tengdu hljóðnema og endurhladdu síðu.")
        else:
            device_options = list(input_devices.keys())

            with st.expander("Hljóðnemi / Microphone", expanded=True):
                device_choice = st.selectbox(
                    "Veldu tæki",
                    options=device_options,
                    label_visibility="collapsed",
                )
                selected_device_id = input_devices[device_choice]

            rec_duration = st.slider(
                "Lengd upptöku (sek.)",
                min_value=1, max_value=60, value=5,
            )

            record_clicked = st.button("Hefja upptöku / Start recording", use_container_width=True, type="primary")

        st.markdown('</div>', unsafe_allow_html=True)

    with results_col:
        if not input_devices:
            pass
        elif record_clicked:
            # --- RECORDING STATE ---
            with st.status("Tek upp… / Recording…", expanded=True) as status:
                st.write(f"Hlusta í {rec_duration} sekúndur…")
                audio_data = audio.record_audio(rec_duration, selected_device_id)
                st.session_state["audio_data"] = audio_data

                # Waveform
                st.markdown('<div class="waveform-label">Hljóðbylgja / Waveform</div>', unsafe_allow_html=True)
                display_n = min(len(audio_data), 4000)
                step = max(1, len(audio_data) // display_n)
                st.line_chart(audio_data[::step], height=100, use_container_width=True)

                # Playback
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, audio_data, 16000)
                    st.audio(tmp.name)
                    tmp_path = tmp.name

                # --- PROCESSING STATE ---
                status.update(label="Umbreyti tal í texta… / Transcribing…")
                st.write("Vinnsla stendur yfir…")
                transcription = transcribe.transcribe_audio(
                    audio_data,
                    st.session_state["model"],
                    st.session_state["processor"],
                    language_token=st.session_state["language_token"],
                )
                st.session_state["last_transcript_text"] = transcription
                st.session_state["gpt_record_response"] = ""
                status.update(label="Tilbúið / Complete", state="complete")

            os.unlink(tmp_path)

            # --- RESULTS ---
            st.markdown(f'<div class="transcript-block">{transcription}</div>', unsafe_allow_html=True)

            col_copy, col_dl = st.columns(2)
            with col_copy:
                st.download_button(
                    "Sækja texta / Download TXT",
                    transcription,
                    file_name="transcript.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with col_dl:
                srt = transcribe.create_srt([f"[0:00:00 → {transcribe.format_timestamp(rec_duration)}] {transcription}"])
                st.download_button(
                    "Sækja SRT / Download SRT",
                    srt,
                    file_name="transcript.srt",
                    mime="text/plain",
                    use_container_width=True,
                )

            render_gpt_assistant(transcription, "gpt_record_response")

        else:
            # --- PLACEHOLDER STATE ---
            st.markdown(
                '<div class="transcript-placeholder">'
                "Textaúttak birtist hér eftir upptöku.<br>"
                "<small>Transcript will appear here after recording.</small>"
                "</div>",
                unsafe_allow_html=True,
            )

# ========== TAB: UPLOAD ====================================================
with tab_upload:
    controls_col2, results_col2 = st.columns([2, 3], gap="large")

    with controls_col2:
        st.markdown('<div class="card"><div class="card-title">Hlaða upp hljóðskrá / Upload audio</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Veldu hljóðskrá",
            type=["wav", "mp3", "m4a", "flac"],
            help="WAV, MP3, M4A, FLAC — hámark 1 GB",
            key="file_uploader",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            st.audio(uploaded_file)

            with st.expander("Stillingar / Settings", expanded=False):
                chunk_size = st.slider(
                    "Lengd hluta (sek.)",
                    min_value=10, max_value=60, value=30,
                    help="Styttri hlutar geta aukið nákvæmni en taka lengri tíma.",
                )
        else:
            chunk_size = 30

        process_clicked = uploaded_file is not None and st.button(
            "Hefja vinnslu / Start processing",
            use_container_width=True,
            type="primary",
            disabled=uploaded_file is None,
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with results_col2:
        if process_clicked and uploaded_file is not None:
            with st.status("Greini skrá… / Analyzing file…", expanded=True) as status:
                audio_data, duration, file_info = audio.load_audio_file(uploaded_file)

                if audio_data is not None and file_info is not None:
                    # File info pills
                    pills = " ".join(f'<span class="file-pill">{k}: {v}</span>' for k, v in file_info.items())
                    st.markdown(pills, unsafe_allow_html=True)

                    num_segments = math.ceil(duration / chunk_size)
                    st.write(f"{num_segments} hlutar · ~{transcribe.format_timestamp(duration)} af hljóði")

                    status.update(label=f"Umbreyti {num_segments} hluta… / Transcribing…")

                    transcriptions = transcribe.transcribe_long_audio(
                        audio_data,
                        st.session_state["model"],
                        st.session_state["processor"],
                        duration,
                        chunk_size,
                        language_token=st.session_state["language_token"],
                    )

                    st.session_state["last_transcript_text"] = "\n".join(transcriptions)
                    st.session_state["gpt_upload_response"] = ""
                    status.update(label="Tilbúið / Complete", state="complete")

            # --- RESULTS ---
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                full_text = "\n\n".join(transcriptions)
                st.download_button(
                    "Sækja texta / Download TXT",
                    full_text,
                    file_name=f"{uploaded_file.name}_transcript.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with dl_col2:
                srt_content = transcribe.create_srt(transcriptions)
                st.download_button(
                    "Sækja SRT / Download SRT",
                    srt_content,
                    file_name=f"{uploaded_file.name}_transcript.srt",
                    mime="text/plain",
                    use_container_width=True,
                )

            for trans in transcriptions:
                st.markdown(f'<div class="transcript-block">{trans}</div>', unsafe_allow_html=True)

            render_gpt_assistant("\n".join(transcriptions), "gpt_upload_response")

        elif uploaded_file is not None:
            st.markdown(
                '<div class="transcript-placeholder">'
                "Smelltu á <strong>Hefja vinnslu</strong> til að umbreyta.<br>"
                "<small>Click <strong>Start processing</strong> to transcribe.</small>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="transcript-placeholder">'
                "Veldu hljóðskrá til að byrja.<br>"
                "<small>Select an audio file to begin.</small>"
                "</div>",
                unsafe_allow_html=True,
            )
