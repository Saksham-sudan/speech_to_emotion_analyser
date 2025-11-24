import streamlit as st
import numpy as np
import pandas as pd
import torch
import queue
import time
import av
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    WhisperProcessor,
    WhisperForConditionalGeneration
)

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="EmoVoice Cloud",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .transcript-box {
        border: 1px solid #4b5563;
        border-radius: 10px;
        padding: 15px;
        background-color: #1f2937;
        color: #e5e7eb;
        font-style: italic;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- UPDATED CONSTANTS ---
# Switch to 'r-f' model which is more stable for standard inference
MODEL_NAME = "r-f/wav2vec-english-speech-emotion-recognition"
ASR_MODEL_NAME = "distil-whisper/distil-small.en"

SAMPLING_RATE = 16000
WINDOW_DURATION = 3
VAD_CHUNK_SIZE = 512 
ALPHA = 0.2 

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_models():
    with st.spinner("Loading & Quantizing Models..."):
        # 1. Emotion Model
        # Use FeatureExtractor to avoid "vocab.json" errors
        processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
        
        # Quantize (Compress) Emotion Model for Cloud RAM
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        model.eval()
        
        # 2. VAD Model
        vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        
        # 3. Transcription Model
        asr_processor = WhisperProcessor.from_pretrained(ASR_MODEL_NAME)
        asr_model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL_NAME)
        
        # Quantize Whisper Model
        asr_model = torch.quantization.quantize_dynamic(
            asr_model, {torch.nn.Linear}, dtype=torch.qint8
        )
        asr_model.eval()

    return processor, model, vad_model, asr_processor, asr_model

processor, model, vad_model, asr_processor, asr_model = load_models()

# --- 3. HELPER FUNCTIONS ---
def transcribe_audio(audio_data):
    input_features = asr_processor(
        audio_data, 
        sampling_rate=SAMPLING_RATE, 
        return_tensors="pt"
    ).input_features
    
    with torch.no_grad():
        predicted_ids = asr_model.generate(input_features, max_new_tokens=128)
    
    transcription = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()

def create_stable_chart(history_df, labels):
    fig = go.Figure()
    for label in labels:
        if label in history_df.columns:
            fig.add_trace(go.Scatter(
                x=history_df.index * 0.5, 
                y=history_df[label],
                mode='lines',
                name=label.capitalize(),
                line=dict(width=2)
            ))
    fig.update_layout(
        xaxis_title="Steps",
        yaxis_title="Confidence",
        yaxis=dict(range=[0, 1.05]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.1)
    )
    return fig

# --- 4. WEBRTC AUDIO PROCESSOR ---
data_queue = queue.Queue()

def process_audio(frame: av.AudioFrame):
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=SAMPLING_RATE,
    )
    frames = resampler.resample(frame)
    converted_samples = frames[0].to_ndarray().flatten()
    converted_samples = converted_samples.astype(np.float32) / 32768.0
    data_queue.put(converted_samples)
    return frame

# --- 5. UI LAYOUT ---
with st.sidebar:
    st.title("⚙️ Settings")
    enable_transcription = st.toggle("📝 Enable Transcription", value=True)
    st.divider()
    st.info(f"Emotion Model: {MODEL_NAME}\nASR Model: {ASR_MODEL_NAME}")

st.title("🎙️ EmoVoice Cloud")
st.markdown("Real-time Emotion & Speech Analysis")

if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []

m1, m2, m3 = st.columns(3)
with m1: metric_emotion = st.empty()
with m2: metric_confidence = st.empty()
with m3: metric_vad = st.empty()

st.divider()

col_viz, col_text = st.columns([2, 1])
with col_viz:
    st.subheader("Emotion Timeline")
    chart_placeholder = st.empty()
with col_text:
    st.subheader("Live Transcript")
    transcription_placeholder = st.empty()

st.markdown("### 🔴 Start Analysis")
st.caption("Click START below. Allow browser microphone access.")

ctx = webrtc_streamer(
    key="emotion-ai",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": False, "audio": True},
    audio_frame_callback=process_audio,
)

if ctx.state.playing:
    audio_buffer = np.zeros(int(WINDOW_DURATION * SAMPLING_RATE), dtype='float32')
    speech_buffer = []
    
    metric_emotion.metric("Dominant Emotion", "Waiting...")
    metric_confidence.metric("Confidence", "0%")
    metric_vad.metric("Voice Status", "Silence")
    
    num_labels = len(model.config.id2label)
    smoothed_probs = np.zeros(num_labels, dtype='float32')
    uniform_probs = np.full(num_labels, 1.0 / num_labels, dtype='float32')
    
    full_transcript = ""
    buffer_fill_level = 0
    
    while ctx.state.playing:
        try:
            chunks = []
            while True:
                try:
                    chunk = data_queue.get_nowait()
                    chunks.append(chunk)
                except queue.Empty:
                    break
            
            if not chunks:
                time.sleep(0.05)
                continue
                
            new_data = np.concatenate(chunks)
            overlap_len = len(new_data)
            audio_buffer = np.roll(audio_buffer, -overlap_len)
            audio_buffer[-overlap_len:] = new_data
            buffer_fill_level += overlap_len
            
            if buffer_fill_level < SAMPLING_RATE:
                metric_vad.metric("Status", "Buffering...")
                continue

            # VAD Check
            vad_window = audio_buffer[-int(0.5 * SAMPLING_RATE):]
            vad_tensor = torch.tensor(vad_window)
            
            has_speech = False
            vad_chunks = len(vad_window) // VAD_CHUNK_SIZE
            for i in range(vad_chunks):
                chunk_t = vad_tensor[i*VAD_CHUNK_SIZE:(i+1)*VAD_CHUNK_SIZE]
                if vad_model(chunk_t, SAMPLING_RATE).item() > 0.5:
                    has_speech = True
                    break
            
            if has_speech:
                if enable_transcription: 
                    speech_buffer.append(new_data)
                
                inputs = processor(
                    torch.tensor(audio_buffer), 
                    sampling_rate=SAMPLING_RATE, 
                    return_tensors="pt", 
                    padding=True
                )
                
                with torch.no_grad():
                    logits = model(**inputs).logits
                
                probs = torch.softmax(logits, dim=1)[0].numpy()
                smoothed_probs = ALPHA * probs + (1 - ALPHA) * smoothed_probs
                
            else:
                smoothed_probs = 0.05 * uniform_probs + 0.95 * smoothed_probs
                
                # Transcribe
                if len(speech_buffer) > 0 and enable_transcription:
                    total_speech = np.concatenate(speech_buffer)
                    if len(total_speech) > int(0.8 * SAMPLING_RATE): 
                        text = transcribe_audio(total_speech)
                        if text:
                            full_transcript += f" {text}"
                            display = "..." + full_transcript[-100:] if len(full_transcript) > 100 else full_transcript
                            transcription_placeholder.markdown(f"<div class='transcript-box'>{display}</div>", unsafe_allow_html=True)
                    speech_buffer = []

            st.session_state.emotion_history.append(smoothed_probs)
            
            top_id = np.argmax(smoothed_probs)
            top_lbl = model.config.id2label[top_id].capitalize()
            top_conf = smoothed_probs[top_id]
            
            metric_emotion.metric("Dominant Emotion", top_lbl)
            metric_confidence.metric("Confidence", f"{top_conf:.0%}")
            metric_vad.metric("Voice Status", "🗣️ Speaking" if has_speech else "🤫 Silence")
            
            if len(st.session_state.emotion_history) % 5 == 0:
                df = pd.DataFrame(st.session_state.emotion_history, columns=list(model.config.id2label.values()))
                if len(df) > 60: df = df.tail(60)
                chart_placeholder.plotly_chart(
                    create_stable_chart(df, list(model.config.id2label.values())), 
                    use_container_width=True,
                    key=f"chart_{len(st.session_state.emotion_history)}"
                )
                
        except Exception as e:
            print(f"Error: {e}")
            break
