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
# Check transformers version and import accordingly
import transformers
print(f"Transformers version: {transformers.__version__}")

if transformers.__version__ < "4.0.0":
    st.error(f"""
    ⚠️ **Your transformers library is too old (version {transformers.__version__})**
    
    Please update it by running:
    ```
    pip install --upgrade transformers==4.36.0
    ```
    """)
    st.stop()

try:
    from transformers import Wav2Vec2ForSequenceClassification
except ImportError:
    try:
        from transformers import Wav2Vec2ForAudioClassification as Wav2Vec2ForSequenceClassification
    except ImportError:
        st.error("""
        ❌ **Cannot import required Wav2Vec2 model class**
        
        Please run these commands:
        ```bash
        pip uninstall transformers -y
        pip install transformers==4.36.0
        pip install torch torchvision torchaudio
        ```
        """)
        st.stop()

from transformers import (
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

# Enhanced Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1d3a 100%);
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Emotion badges with colors */
    .emotion-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
        margin: 4px;
    }
    
    .emotion-happy { background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%); color: #000; }
    .emotion-sad { background: linear-gradient(135deg, #4a90e2 0%, #67b5ff 100%); color: #fff; }
    .emotion-angry { background: linear-gradient(135deg, #ff4757 0%, #ff6b81 100%); color: #fff; }
    .emotion-neutral { background: linear-gradient(135deg, #95a5a6 0%, #b2bec3 100%); color: #fff; }
    .emotion-fear { background: linear-gradient(135deg, #8e44ad 0%, #9b59b6 100%); color: #fff; }
    
    /* Improved buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 12px 28px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Transcript styling */
    .transcript-box {
        background: rgba(30, 30, 60, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 4px solid #667eea;
        border-radius: 12px;
        padding: 20px;
        color: #e5e7eb;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .transcript-segment {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    
    /* Metrics enhancement */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Pulse animation for active status */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .status-active {
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Control panel styling */
    .control-panel {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 24px;
    }
    
    /* Status card */
    .status-card {
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Constants
MODEL_NAME = "r-f/wav2vec-english-speech-emotion-recognition"
ASR_MODEL_NAME = "openai/whisper-tiny" 
SAMPLING_RATE = 16000
WINDOW_DURATION = 3
STEP_DURATION = 0.5
SAMPLES_PER_STEP = int(STEP_DURATION * SAMPLING_RATE)
WINDOW_SAMPLES = int(WINDOW_DURATION * SAMPLING_RATE)
CONFIDENCE_THRESHOLD = 0.05
ALPHA = 0.2
VAD_CHUNK_SIZE = 512

# Emotion color mapping
EMOTION_COLORS = {
    'happy': '#ffd700',
    'sad': '#4a90e2',
    'angry': '#ff4757',
    'neutral': '#95a5a6',
    'fear': '#8e44ad',
    'disgust': '#e67e22',
    'surprise': '#1abc9c'
}

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
        predicted_ids = asr_model.generate(input_features)
    transcription = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()

def hex_to_rgba(hex_color, alpha=0.2):
    """Convert hex color to rgba format"""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'

def create_advanced_chart(history_df, labels, chart_type="line"):
    fig = go.Figure()
    
    if chart_type == "line":
        for idx, label in enumerate(labels):
            if label in history_df.columns:
                color = EMOTION_COLORS.get(label.lower(), '#667eea')
                fig.add_trace(go.Scatter(
                    x=history_df.index * 0.5,
                    y=history_df[label],
                    mode='lines',
                    name=label.capitalize(),
                    line=dict(width=3, color=color),
                    fill='tozeroy',
                    fillcolor=hex_to_rgba(color, 0.2)
                ))
    
    elif chart_type == "stacked":
        for label in labels:
            if label in history_df.columns:
                color = EMOTION_COLORS.get(label.lower(), '#667eea')
                fig.add_trace(go.Scatter(
                    x=history_df.index * 0.5,
                    y=history_df[label],
                    mode='lines',
                    name=label.capitalize(),
                    stackgroup='one',
                    fillcolor=color
                ))
    
    elif chart_type == "heatmap":
        fig = go.Figure(data=go.Heatmap(
            z=history_df[labels].T.values,
            x=history_df.index * 0.5,
            y=[l.capitalize() for l in labels],
            colorscale='Viridis',
            showscale=True
        ))
    
    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Confidence" if chart_type != "heatmap" else None,
        yaxis=dict(range=[0, 1.05]) if chart_type != "heatmap" else None,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white", size=12),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
        hovermode='x unified',
        height=400
    )
    
    return fig

def render_status_card(title, value, icon, color="#667eea", pulsing=False):
    pulse_class = "status-active" if pulsing else ""
    return f"""
    <div style='background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
                border-left: 4px solid {color};
                border-radius: 12px;
                padding: 16px;
                text-align: center;'>
        <div style='font-size: 2em; margin-bottom: 8px;' class='{pulse_class}'>{icon}</div>
        <div style='font-size: 0.85em; color: #9ca3af; margin-bottom: 4px;'>{title}</div>
        <div style='font-size: 1.4em; font-weight: 700; color: white;'>{value}</div>
    </div>
    """

def render_dominant_emotion_card(emotion, confidence):
    color = EMOTION_COLORS.get(emotion.lower(), '#667eea')
    emoji_map = {
        'happy': '😊', 'sad': '😢', 'angry': '😠', 
        'neutral': '😐', 'fear': '😨', 'disgust': '🤢', 'surprise': '😲'
    }
    emoji = emoji_map.get(emotion.lower(), '🎭')
    
    return f"""
    <div style='background: linear-gradient(135deg, {color}33 0%, {color}22 100%);
                border-radius: 16px; padding: 20px; text-align: center; 
                border: 2px solid {color}66;'>
        <div style='font-size: 0.9em; color: #9ca3af; margin-bottom: 8px;'>DOMINANT EMOTION</div>
        <div style='font-size: 2.5em; font-weight: 800; margin: 10px 0;'>{emoji} {emotion}</div>
        <div style='font-size: 1.2em; color: {color};'>Confidence: {confidence:.0%}</div>
    </div>
    """

# --- 4. UI LAYOUT ---

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("Configure your analysis session.")
    enable_transcription = st.toggle("📝 Enable Transcription", value=True)
    chart_type = st.selectbox("Chart Style", ["Line", "Stacked", "Heatmap"], index=0)
    st.divider()
    st.info("Uses **Wav2Vec2** for emotions and **Whisper** for text. Processing happens locally.")
    
    # Additional settings
    st.markdown("### Advanced")
    show_confidence_threshold = st.slider("Confidence Display Threshold", 0.0, 1.0, 0.3, 0.05)
    
# Main Header
st.title("🎙️ EmoVoice Analytics")
st.markdown("Real-time emotional intelligence dashboard with advanced visualization.")

# State Setup
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'transcript_segments' not in st.session_state:
    st.session_state.transcript_segments = []
if 'session_start_time' not in st.session_state:
    st.session_state.session_start_time = None
    
data_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status: print(status, flush=True)
    data_queue.put(indata.copy())

# Tabs for Modes
tab_live, tab_file, tab_report = st.tabs(["🔴 Live Analysis", "📂 File Upload", "📊 Session Report"])

# --- TAB 1: LIVE ANALYSIS ---
with tab_live:
    # Control Panel
    st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
    
    control_col1, control_col2, control_col3, control_col4 = st.columns([2, 1, 1, 1])
    
    with control_col1:
        run_button = st.toggle("🔴 Start Recording", key="live_btn")
    
    with control_col2:
        if st.button("💾 Save Session"):
            if len(st.session_state.emotion_history) > 0:
                st.success("Session data ready for download in Report tab!")
    
    with control_col3:
        if st.button("🗑️ Clear Data"):
            st.session_state.emotion_history = []
            st.session_state.transcript_segments = []
            st.rerun()
    
    with control_col4:
        if run_button:
            st.markdown("<span style='color: #10b981; font-weight: 600;'>● LIVE</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color: #6b7280; font-weight: 600;'>○ STANDBY</span>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Enhanced HUD (Heads Up Display)
    hud_col1, hud_col2, hud_col3, hud_col4 = st.columns([3, 1, 1, 1])
    
    with hud_col1:
        dominant_emotion_placeholder = st.empty()
    
    with hud_col2:
        status_vad_placeholder = st.empty()
    
    with hud_col3:
        status_timer_placeholder = st.empty()
    
    with hud_col4:
        status_samples_placeholder = st.empty()

    st.divider()

    # Chart & Transcript Area
    col_viz, col_text = st.columns([2, 1])
    
    with col_viz:
        st.subheader("📈 Emotion Timeline")
        chart_placeholder = st.empty()
        
    with col_text:
        st.subheader("📝 Live Transcript")
        transcription_placeholder = st.empty()
        
    # --- LOGIC LOOP ---
    if run_button:
        if st.session_state.session_start_time is None:
            st.session_state.session_start_time = time.time()
            
        audio_buffer = np.zeros(WINDOW_SAMPLES, dtype='float32')
        stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLING_RATE, blocksize=SAMPLES_PER_STEP)
        stream.start()
        
        speech_buffer = []
        
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
                    warmup_pct = int((current_chunks/warmup_chunks_needed)*100)
                    status_vad_placeholder.markdown(
                        render_status_card("System", f"Warming {warmup_pct}%", "⏳", "#fbbf24"),
                        unsafe_allow_html=True
                    )
                    if current_chunks >= warmup_chunks_needed: 
                        buffer_is_warm = True
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
                        text = transcribe_audio(np.concatenate(speech_buffer))
                        if text:
                            current_time = time.time() - st.session_state.session_start_time
                            top_emotion = model.config.id2label[np.argmax(smoothed_probs)]
                            
                            st.session_state.transcript_segments.append({
                                'time': current_time,
                                'text': text,
                                'emotion': top_emotion
                            })
                            full_transcript += f" {text}"
                            
                            # Render transcript with emotion colors
                            transcript_html = "<div style='max-height: 400px; overflow-y: auto;'>"
                            for seg in st.session_state.transcript_segments[-5:]:
                                color = EMOTION_COLORS.get(seg['emotion'].lower(), '#667eea')
                                transcript_html += f"""
                                <div class='transcript-segment' style='border-left: 3px solid {color};'>
                                    <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                                        <span style='color: {color}; font-weight: 600;'>{seg['emotion'].capitalize()}</span>
                                        <span style='color: #6b7280; font-size: 0.85em;'>{seg['time']:.1f}s</span>
                                    </div>
                                    <div style='color: #e5e7eb;'>{seg['text']}</div>
                                </div>
                                """
                            transcript_html += "</div>"
                            transcription_placeholder.markdown(transcript_html, unsafe_allow_html=True)
                        speech_buffer = []

                # 5. UI Updates
                st.session_state.emotion_history.append(smoothed_probs)
                
                # Update Dominant Emotion Card
                top_id = np.argmax(smoothed_probs)
                top_lbl = model.config.id2label[top_id].capitalize()
                top_conf = smoothed_probs[top_id]
                
                dominant_emotion_placeholder.markdown(
                    render_dominant_emotion_card(top_lbl, top_conf),
                    unsafe_allow_html=True
                )
                
                # Update Status Cards
                status_vad_placeholder.markdown(
                    render_status_card(
                        "Voice Activity", 
                        "Speaking" if has_speech else "Silence",
                        "🗣️" if has_speech else "🤫",
                        "#10b981" if has_speech else "#6b7280",
                        pulsing=has_speech
                    ),
                    unsafe_allow_html=True
                )
                
                # Timer
                elapsed = time.time() - st.session_state.session_start_time
                status_timer_placeholder.markdown(
                    render_status_card("Duration", f"{elapsed:.1f}s", "⏱️", "#8b5cf6"),
                    unsafe_allow_html=True
                )
                
                # Samples
                status_samples_placeholder.markdown(
                    render_status_card("Samples", str(len(st.session_state.emotion_history)), "📊", "#06b6d4"),
                    unsafe_allow_html=True
                )
                
                # Update Chart (Every 2 steps for performance)
                if len(st.session_state.emotion_history) % 2 == 0:
                    df = pd.DataFrame(st.session_state.emotion_history, columns=list(model.config.id2label.values()))
                    if len(df) > 60: df = df.tail(60)
                    
                    fig = create_advanced_chart(df, list(model.config.id2label.values()), chart_type.lower())
                    chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"live_{len(st.session_state.emotion_history)}")

            except queue.Empty:
                time.sleep(0.05)
                continue
        
        stream.stop()
        stream.close()

# --- TAB 2: FILE UPLOAD ---
with tab_file:
    st.markdown("### Upload & Analyze Audio Files")
    
    uploaded_file = st.file_uploader("Choose an audio file (WAV/MP3)", type=["wav", "mp3"])
    
    if uploaded_file and st.button("🔍 Analyze File", type="primary"):
        with st.status("Processing File...", expanded=True) as status:
            st.write("📂 Loading audio...")
            y, sr = librosa.load(uploaded_file, sr=SAMPLING_RATE)
            
            st.write("🎵 Reducing noise...")
            y_clean = nr.reduce_noise(y=y, sr=SAMPLING_RATE)
            
            st.write("🧠 Analyzing emotions & extracting text...")
            
            st.session_state.emotion_history = []
            st.session_state.transcript_segments = []
            audio_buffer = np.zeros(WINDOW_SAMPLES, dtype='float32')
            num_labels = len(model.config.id2label)
            smoothed_probs = np.zeros(num_labels, dtype='float32')
            uniform_probs = np.full(num_labels, 1.0 / num_labels, dtype='float32')
            file_speech_buffer = []
            
            total_chunks = len(y_clean) // SAMPLES_PER_STEP
            prog = st.progress(0)
            
            warmed = False
            cur_chunks = 0
            warmup_chunks_needed = WINDOW_SAMPLES // SAMPLES_PER_STEP
            
            for i in range(total_chunks):
                chunk = y_clean[i*SAMPLES_PER_STEP:(i+1)*SAMPLES_PER_STEP]
                if len(chunk) < SAMPLES_PER_STEP: 
                    chunk = np.pad(chunk, (0, SAMPLES_PER_STEP-len(chunk)))
                
                audio_buffer = np.roll(audio_buffer, -len(chunk))
                audio_buffer[-len(chunk):] = chunk
                
                if not warmed:
                    cur_chunks += 1
                    if cur_chunks >= warmup_chunks_needed: warmed = True
                    continue
                
                # VAD
                has_speech = False
                num_vad = len(chunk) // VAD_CHUNK_SIZE
                for j in range(num_vad):
                    if vad_model(torch.tensor(chunk[j*VAD_CHUNK_SIZE:(j+1)*VAD_CHUNK_SIZE]), SAMPLING_RATE).item() > 0.5:
                        has_speech = True
                        break
                
                if has_speech:
                    if enable_transcription: file_speech_buffer.append(chunk)
                    inputs = processor(torch.tensor(audio_buffer), sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
                    with torch.no_grad(): 
                        logits = model(**inputs).logits
                    smoothed_probs = ALPHA * torch.softmax(logits, dim=1)[0].numpy() + (1-ALPHA)*smoothed_probs
                else:
                    smoothed_probs = 0.05 * uniform_probs + 0.95 * smoothed_probs

                st.session_state.emotion_history.append(smoothed_probs)
                prog.progress((i+1)/total_chunks)
            
            status.update(label="✅ Analysis Complete!", state="complete")
        
        # Transcribe full file
        if enable_transcription and len(file_speech_buffer) > 0:
            with st.spinner("🎯 Generating full transcript..."):
                final_text = transcribe_audio(np.concatenate(file_speech_buffer))
                st.session_state.transcript_segments.append({
                    'time': 0,
                    'text': final_text,
                    'emotion': 'neutral'
                })
        
        # Show Results
        st.success("Analysis complete! View results below.")
        
        result_col1, result_col2 = st.columns([2, 1])
        
        with result_col1:
            if len(st.session_state.emotion_history) > 0:
                st.subheader("📈 Emotion Timeline")
                df = pd.DataFrame(st.session_state.emotion_history, columns=list(model.config.id2label.values()))
                fig = create_advanced_chart(df, list(model.config.id2label.values()), chart_type.lower())
                st.plotly_chart(fig, use_container_width=True)
        
        with result_col2:
            if st.session_state.transcript_segments:
                st.subheader("📝 Transcript")
                for seg in st.session_state.transcript_segments:
                    color = EMOTION_COLORS.get(seg['emotion'].lower(), '#667eea')
                    st.markdown(f"""
                    <div class='transcript-segment' style='border-left: 3px solid {color};'>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                            <span style='color: {color}; font-weight: 600;'>{seg['emotion'].capitalize()}</span>
                            <span style='color: #6b7280; font-size: 0.85em;'>{seg['time']:.1f}s</span>
                        </div>
                        <div style='color: #e5e7eb;'>{seg['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)

# --- TAB 3: ENHANCED REPORT ---
with tab_report:
    st.markdown("### 📊 Session Analytics & Export")
    
    if len(st.session_state.emotion_history) > 0:
        df = pd.DataFrame(st.session_state.emotion_history, 
                         columns=list(model.config.id2label.values()))
        df['Time'] = df.index * STEP_DURATION
        
        # Summary Stats Row
        st.markdown("#### 📈 Session Summary")
        
        stat1, stat2, stat3, stat4, stat5 = st.columns(5)
        
        with stat1:
            total_time = len(df) * STEP_DURATION
            st.metric("Duration", f"{total_time:.1f}s", help="Total recording duration")
            
        with stat2:
            dominant = df.drop('Time', axis=1).idxmax(axis=1).mode()[0]
            st.metric("Most Frequent", dominant.capitalize(), help="Most common emotion")
            
        with stat3:
            avg_conf = df.drop('Time', axis=1).max(axis=1).mean()
            st.metric("Avg Confidence", f"{avg_conf:.1%}", help="Average confidence score")
            
        with stat4:
            emotion_switches = (df.drop('Time', axis=1).idxmax(axis=1) != df.drop('Time', axis=1).idxmax(axis=1).shift()).sum()
            st.metric("Emotion Switches", emotion_switches, help="Number of emotion changes")
            
        with stat5:
            samples = len(df)
            st.metric("Total Samples", samples, help="Number of analyzed samples")
        
        st.divider()
        
        # Visualizations
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.markdown("#### 🥧 Emotion Distribution")
            emotion_cols = [col for col in df.columns if col != 'Time']
            emotion_totals = df[emotion_cols].sum()
            
            colors_list = [EMOTION_COLORS.get(e.lower(), '#667eea') for e in emotion_totals.index]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=[l.capitalize() for l in emotion_totals.index],
                values=emotion_totals.values,
                hole=0.4,
                marker=dict(colors=colors_list)
            )])
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="white"),
                height=300,
                showlegend=True
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with viz_col2:
            st.markdown("#### 📊 Peak Confidence by Emotion")
            max_conf = df[emotion_cols].max()
            
            colors_list = [EMOTION_COLORS.get(e.lower(), '#667eea') for e in max_conf.index]
            
            fig_bar = go.Figure(data=[go.Bar(
                x=[l.capitalize() for l in max_conf.index],
                y=max_conf.values,
                marker_color=colors_list
            )])
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="white"),
                yaxis=dict(range=[0, 1], title="Confidence"),
                xaxis=dict(title="Emotion"),
                height=300
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.divider()
        
        # Export Options
        st.markdown("#### 💾 Export Options")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV",
                csv,
                "emotion_data.csv",
                "text/csv",
                use_container_width=True
            )
        
        with export_col2:
            # JSON export with transcript
            import json
            export_data = {
                'emotions': df.to_dict(orient='records'),
                'transcripts': st.session_state.transcript_segments,
                'summary': {
                    'duration': total_time,
                    'dominant_emotion': dominant,
                    'avg_confidence': float(avg_conf),
                    'emotion_switches': int(emotion_switches)
                }
            }
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                "📦 Download JSON",
                json_str,
                "emotion_analysis.json",
                "application/json",
                use_container_width=True
            )
        
        with export_col3:
            if st.button("🗑️ Clear All Data", use_container_width=True, type="secondary"):
                st.session_state.emotion_history = []
                st.session_state.transcript_segments = []
                st.session_state.session_start_time = None
                st.rerun()
        
        st.divider()
        
        # Data Table
        st.markdown("#### 📋 Raw Data Preview")
        st.dataframe(
            df.head(100), 
            use_container_width=True,
            height=400
        )
        
    else:
        st.info("📭 No data recorded yet. Go to 'Live Analysis' or 'File Upload' to generate data.")
        st.markdown("""
        <div style='text-align: center; padding: 40px;'>
            <div style='font-size: 4em; margin-bottom: 20px;'>🎙️</div>
            <p style='color: #9ca3af; font-size: 1.1em;'>Start recording or upload a file to see analytics here</p>
        </div>
        """, unsafe_allow_html=True)
