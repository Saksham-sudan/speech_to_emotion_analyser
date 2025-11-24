import streamlit as st
import sounddevice as sd
import noisereduce as nr
import numpy as np
import pandas as pd
import librosa
import torch
import time
import queue
import plotly.graph_objects as go
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    WhisperProcessor,
    WhisperForConditionalGeneration
)

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="EmoVoice Analytics",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff6b6b;
    }
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

# Constants
MODEL_NAME = "harshit345/xlsr-wav2vec-speech-emotion-recognition"
ASR_MODEL_NAME = "distil-whisper/distil-small.en"
SAMPLING_RATE = 16000
WINDOW_DURATION = 3
STEP_DURATION = 0.5
SAMPLES_PER_STEP = int(STEP_DURATION * SAMPLING_RATE)
WINDOW_SAMPLES = int(WINDOW_DURATION * SAMPLING_RATE)
CONFIDENCE_THRESHOLD = 0.05
ALPHA = 0.2
VAD_CHUNK_SIZE = 512

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_models():
    with st.spinner("Loading AI Models (Wav2Vec2 + Whisper)..."):
        # Emotion
        try:
            processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        except Exception:
            processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()
        
        # VAD
        vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        
        # Transcription
        asr_processor = WhisperProcessor.from_pretrained(ASR_MODEL_NAME)
        asr_model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL_NAME)
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
        predicted_ids = asr_model.generate(input_features, max_new_tokens=128, forced_decoder_ids=None)
    transcription = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()

def create_stable_chart(history_df, labels):
    # Create a Plotly figure that doesn't jump around
    fig = go.Figure()
    
    # Add a line for each emotion
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
        xaxis_title="Time (seconds)",
        yaxis_title="Confidence",
        yaxis=dict(range=[0, 1.05]), # Fixed Y-axis for stability
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.1)
    )
    return fig

# --- 4. UI LAYOUT ---

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("Configure your analysis session.")
    enable_transcription = st.toggle("📝 Enable Transcription", value=True)
    st.divider()
    st.info("Uses **Wav2Vec2** for emotions and **Whisper** for text. Processing happens locally.")

# Main Header
st.title("🎙️ EmoVoice Analytics")
st.markdown("Real-time emotional intelligence dashboard.")

# State Setup
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
data_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status: print(status, flush=True)
    data_queue.put(indata.copy())

# Tabs for Modes
tab_live, tab_file, tab_report = st.tabs(["🔴 Live Analysis", "📂 File Upload", "📊 Session Report"])

# --- TAB 1: LIVE ANALYSIS ---
with tab_live:
    col_start, col_status = st.columns([1, 3])
    with col_start:
        run_button = st.toggle("Start Recording", key="live_btn")
    with col_status:
        if run_button:
            st.success("System Active & Listening...")
        else:
            st.warning("System Standby")

    # HUD (Heads Up Display)
    m1, m2, m3 = st.columns(3)
    with m1:
        metric_emotion = st.empty()
    with m2:
        metric_confidence = st.empty()
    with m3:
        metric_vad = st.empty()

    st.divider()

    # Chart & Transcript Area
    col_viz, col_text = st.columns([2, 1])
    
    with col_viz:
        st.subheader("Emotion Timeline")
        chart_placeholder = st.empty()
        
    with col_text:
        st.subheader("Live Transcript")
        transcription_placeholder = st.empty()
        
    # --- LOGIC LOOP ---
    if run_button:
        audio_buffer = np.zeros(WINDOW_SAMPLES, dtype='float32')
        stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLING_RATE, blocksize=SAMPLES_PER_STEP)
        stream.start()
        
        st.session_state.emotion_history = []
        speech_buffer = []
        
        # Init Placeholders
        metric_emotion.metric("Dominant Emotion", "Waiting...")
        metric_confidence.metric("Confidence", "0%")
        metric_vad.metric("Voice Status", "Silence")
        transcription_placeholder.markdown("<div class='transcript-box'>...</div>", unsafe_allow_html=True)
        
        # Loop Vars
        num_labels = len(model.config.id2label)
        smoothed_probs = np.zeros(num_labels, dtype='float32')
        uniform_probs = np.full(num_labels, 1.0 / num_labels, dtype='float32')
        buffer_is_warm = False
        current_chunks = 0
        warmup_chunks_needed = WINDOW_SAMPLES // SAMPLES_PER_STEP
        full_transcript = ""

        while run_button:
            try:
                # 1. Get Audio
                raw_chunk = data_queue.get(timeout=0.1).squeeze()
                clean_chunk = nr.reduce_noise(y=raw_chunk, sr=SAMPLING_RATE)
                audio_buffer = np.roll(audio_buffer, -len(clean_chunk))
                audio_buffer[-len(clean_chunk):] = clean_chunk
                
                # 2. Warmup
                if not buffer_is_warm:
                    current_chunks += 1
                    metric_vad.metric("Voice Status", f"Warming Up {int((current_chunks/warmup_chunks_needed)*100)}%")
                    if current_chunks >= warmup_chunks_needed: buffer_is_warm = True
                    continue

                # 3. VAD
                has_speech = False
                num_vad_chunks = len(clean_chunk) // VAD_CHUNK_SIZE
                for i in range(num_vad_chunks):
                    vad_tensor = torch.tensor(clean_chunk[i*VAD_CHUNK_SIZE:(i+1)*VAD_CHUNK_SIZE])
                    if vad_model(vad_tensor, SAMPLING_RATE).item() > 0.5:
                        has_speech = True
                        break

                # 4. Processing
                if has_speech:
                    if enable_transcription: speech_buffer.append(raw_chunk)
                    
                    audio_tensor = torch.tensor(audio_buffer)
                    inputs = processor(audio_tensor, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
                    with torch.no_grad():
                        logits = model(**inputs).logits
                    current_probs = torch.softmax(logits, dim=1)[0].numpy()
                    smoothed_probs = ALPHA * current_probs + (1 - ALPHA) * smoothed_probs
                else:
                    smoothed_probs = 0.05 * uniform_probs + 0.95 * smoothed_probs
                    
                    # Transcribe on Silence
                    if len(speech_buffer) > 0 and enable_transcription:
                        with st.spinner("Processing text..."):
                            text = transcribe_audio(np.concatenate(speech_buffer))
                            if text:
                                full_transcript += f" {text}"
                                # Keep only last 200 chars for clean UI
                                display_text = "..." + full_transcript[-200:] if len(full_transcript) > 200 else full_transcript
                                transcription_placeholder.markdown(f"<div class='transcript-box'>{display_text}</div>", unsafe_allow_html=True)
                        speech_buffer = []

                # 5. UI Updates
                st.session_state.emotion_history.append(smoothed_probs)
                
                # Update Metrics
                top_id = np.argmax(smoothed_probs)
                top_lbl = model.config.id2label[top_id].capitalize()
                top_conf = smoothed_probs[top_id]
                
                metric_emotion.metric("Dominant Emotion", top_lbl)
                metric_confidence.metric("Confidence", f"{top_conf:.0%}")
                metric_vad.metric("Voice Status", "🗣️ Speaking" if has_speech else "🤫 Silence")
                
                # Update Chart (Every 2 steps for performance)
                if len(st.session_state.emotion_history) % 2 == 0:
                    df = pd.DataFrame(st.session_state.emotion_history, columns=list(model.config.id2label.values()))
                    # Trim for speed
                    if len(df) > 60: df = df.tail(60)
                    
                    fig = create_stable_chart(df, list(model.config.id2label.values()))
                    chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"live_{len(st.session_state.emotion_history)}")

            except queue.Empty:
                time.sleep(0.05)
                continue
        
        # Stop Clean up
        if len(speech_buffer) > 0 and enable_transcription:
            text = transcribe_audio(np.concatenate(speech_buffer))
            if text:
                full_transcript += f" {text}"
                transcription_placeholder.markdown(f"<div class='transcript-box'>{full_transcript}</div>", unsafe_allow_html=True)
                
        stream.stop()
        stream.close()

# --- TAB 2: FILE UPLOAD ---
with tab_file:
    uploaded_file = st.file_uploader("Upload Audio (WAV/MP3)", type=["wav", "mp3"])
    
    if uploaded_file and st.button("Analyze File"):
        with st.status("Processing File...", expanded=True) as status:
            st.write("Loading audio...")
            y, sr = librosa.load(uploaded_file, sr=SAMPLING_RATE)
            st.write("Reducing noise...")
            y_clean = nr.reduce_noise(y=y, sr=SAMPLING_RATE)
            
            st.write("Analyzing emotions & text...")
            # (Simplified loop for file similar to previous code)
            st.session_state.emotion_history = []
            audio_buffer = np.zeros(WINDOW_SAMPLES, dtype='float32')
            smoothed_probs = np.zeros(num_labels, dtype='float32')
            file_speech_buffer = []
            
            total_chunks = len(y) // SAMPLES_PER_STEP
            prog = st.progress(0)
            
            # Warmup logic
            warmed = False
            cur_chunks = 0
            
            for i in range(total_chunks):
                chunk = y_clean[i*SAMPLES_PER_STEP:(i+1)*SAMPLES_PER_STEP]
                if len(chunk) < SAMPLES_PER_STEP: chunk = np.pad(chunk, (0, SAMPLES_PER_STEP-len(chunk)))
                
                audio_buffer = np.roll(audio_buffer, -len(chunk))
                audio_buffer[-len(chunk):] = chunk
                
                if not warmed:
                    cur_chunks += 1
                    if cur_chunks >= warmup_chunks_needed: warmed = True
                    continue
                
                # Check VAD
                has_speech = False
                num_vad = len(chunk) // VAD_CHUNK_SIZE
                for j in range(num_vad):
                    if vad_model(torch.tensor(chunk[j*512:(j+1)*512]), SAMPLING_RATE).item() > 0.5:
                        has_speech = True; break
                
                if has_speech:
                    if enable_transcription: file_speech_buffer.append(chunk)
                    inputs = processor(torch.tensor(audio_buffer), sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
                    with torch.no_grad(): logits = model(**inputs).logits
                    smoothed_probs = ALPHA * torch.softmax(logits, dim=1)[0].numpy() + (1-ALPHA)*smoothed_probs
                else:
                    smoothed_probs = 0.05 * uniform_probs + 0.95 * smoothed_probs
                    # Transcribe intermediate
                    if len(file_speech_buffer) > 0 and enable_transcription:
                         # For file speed, maybe just accumulate and do at end or large chunks
                         pass 

                st.session_state.emotion_history.append(smoothed_probs)
                prog.progress((i+1)/total_chunks)
            
            status.update(label="Analysis Complete!", state="complete")
        
        # Show Results
        if enable_transcription and len(file_speech_buffer) > 0:
            st.subheader("Transcript")
            final_text = transcribe_audio(np.concatenate(file_speech_buffer))
            st.markdown(f"<div class='transcript-box'>{final_text}</div>", unsafe_allow_html=True)
            
        if len(st.session_state.emotion_history) > 0:
            st.subheader("Emotion Timeline")
            df = pd.DataFrame(st.session_state.emotion_history, columns=list(model.config.id2label.values()))
            st.plotly_chart(create_stable_chart(df, list(model.config.id2label.values())), use_container_width=True)

# --- TAB 3: REPORT ---
with tab_report:
    st.markdown("Download the raw data from your latest session.")
    if len(st.session_state.emotion_history) > 0:
        df = pd.DataFrame(st.session_state.emotion_history, columns=list(model.config.id2label.values()))
        df['Time'] = df.index * STEP_DURATION
        
        col_dl, col_clear = st.columns(2)
        with col_dl:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "emotion_data.csv", "text/csv")
        with col_clear:
            if st.button("🗑️ Clear History"):
                st.session_state.emotion_history = []
                st.rerun()
        
        st.dataframe(df.head(50), use_container_width=True)
    else:
        st.info("No data recorded yet. Go to 'Live Analysis' or 'File Upload' to generate data.")
