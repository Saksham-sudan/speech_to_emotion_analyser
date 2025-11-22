import streamlit as st
import sounddevice as sd
import noisereduce as nr
import numpy as np
import pandas as pd
import librosa
import torch
import time
import queue
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC
)

MODEL_NAME = "r-f/wav2vec-english-speech-emotion-recognition"
ASR_MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLING_RATE = 16000
WINDOW_DURATION = 3
STEP_DURATION = 0.5
SAMPLES_PER_STEP = int(STEP_DURATION * SAMPLING_RATE)
WINDOW_SAMPLES = int(WINDOW_DURATION * SAMPLING_RATE)
CONFIDENCE_THRESHOLD = 0.05
ALPHA = 0.2
VAD_CHUNK_SIZE = 512

@st.cache_resource
def load_models():
    try:
        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    except Exception:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    vad_model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )
    asr_processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL_NAME)
    asr_model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL_NAME)
    asr_model.eval()

    return processor, model, vad_model, asr_processor, asr_model

processor, model, vad_model, asr_processor, asr_model = load_models()

st.set_page_config(page_title="🎙️ Real-Time Speech Emotion Analyzer", layout="centered")
st.title("🎙️ Real-Time Speech Emotion Analyzer")
st.markdown("This app uses a **sliding window**, **noise reduction**, **VAD**, and **smoothing** for stable, real-time emotion detection.")

mode = st.radio(
    "Choose analysis mode:",
    ("🎙️ Live", "📂 File"),
    horizontal=True
)

emotion_placeholder = st.empty()
confidence_placeholder = st.empty()

st.markdown("---") 
st.markdown("### Top 3 Emotions:")
top1_placeholder = st.empty()
top2_placeholder = st.empty()
top3_placeholder = st.empty()
st.markdown("---")
st.markdown("### Emotion Timeline")
chart_placeholder = st.empty()
st.markdown("---")
st.markdown("### Transcription")
transcription_placeholder = st.empty()


data_queue = queue.Queue()
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []

num_labels = len(model.config.id2label)
smoothed_probs = np.zeros(num_labels, dtype='float32')

def audio_callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    data_queue.put(indata.copy())

if mode == "🎙️ Live":
    run_button = st.toggle("🎧 Start / Stop Listening")

    if run_button:
        st.info("Listening... Speak now 🎤")
        audio_buffer = np.zeros(WINDOW_SAMPLES, dtype='float32')
    
        stream = sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=SAMPLING_RATE,
            blocksize=SAMPLES_PER_STEP
        )
        stream.start()
        st.session_state.emotion_history = []
        speech_buffer = []

        uniform_probs = np.full(num_labels, 1.0 / num_labels, dtype='float32')
        buffer_is_warm = False
        warmup_chunks_needed = WINDOW_SAMPLES // SAMPLES_PER_STEP
        current_chunks = 0

        while run_button:
            # Initialize has_speech to False at the start of every loop
            has_speech = False
            
            try:
                chunk = data_queue.get(timeout=0.1).squeeze()
                chunk = nr.reduce_noise(y=chunk, sr=SAMPLING_RATE)
                
                audio_buffer = np.roll(audio_buffer, -len(chunk))
                audio_buffer[-len(chunk):] = chunk
                
                if not buffer_is_warm:
                    current_chunks += 1
                    emotion_placeholder.markdown("## 🔥 **Warming up buffer...**")
                    confidence_placeholder.progress(current_chunks / warmup_chunks_needed)
                    
                    if current_chunks >= warmup_chunks_needed:
                        buffer_is_warm = True
                    continue

                # Run VAD
                num_vad_chunks = len(chunk) // VAD_CHUNK_SIZE
                for i in range(num_vad_chunks):
                    start = i * VAD_CHUNK_SIZE
                    end = start + VAD_CHUNK_SIZE
                    vad_input_tensor = torch.tensor(chunk[start:end])
                    speech_prob = vad_model(vad_input_tensor, SAMPLING_RATE).item()
                    if speech_prob > 0.5:
                        has_speech = True
                        break
                
                # Add to buffer only if speech is detected
                if has_speech:
                    speech_buffer.append(chunk)

            except queue.Empty:
                # If queue is empty, we just wait. has_speech remains False.
                time.sleep(0.05)
            
            # --- LOGIC BLOCK (Runs even if queue was empty) ---
            if buffer_is_warm:
                
                if has_speech:
                    # Emotion Prediction
                    audio_tensor = torch.tensor(audio_buffer)
                    inputs = processor(audio_tensor, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
                    with torch.no_grad():
                        logits = model(**inputs).logits
                    current_probs = torch.softmax(logits, dim=1)[0].numpy()
                    smoothed_probs = ALPHA * current_probs + (1 - ALPHA) * smoothed_probs
                
                else:
                    # Silence Logic
                    smoothed_probs = 0.05 * uniform_probs + 0.95 * smoothed_probs
                    
                    # --- TRANSCRIPTION LOGIC (Runs on Silence) ---
                    if len(speech_buffer) > 0:
                        with st.spinner("Transcribing..."):
                            speech_audio = np.concatenate(speech_buffer)
                            speech_buffer = []
                            
                            input_values = asr_processor(
                                speech_audio, 
                                sampling_rate=SAMPLING_RATE, 
                                return_tensors="pt"
                            ).input_values
                            
                            with torch.no_grad():
                                logits = asr_model(input_values).logits
                            
                            predicted_ids = torch.argmax(logits, dim=-1)
                            transcription = asr_processor.batch_decode(predicted_ids)[0]
                            
                            if len(transcription) > 0:
                                transcription_placeholder.markdown(f"**You said:** *{transcription.lower()}*")

                # Update UI
                st.session_state.emotion_history.append(smoothed_probs)
                
                top_3_indices = np.argsort(smoothed_probs)[-3:][::-1]
                top_emotion = model.config.id2label[top_3_indices[0]]
                top_confidence = smoothed_probs[top_3_indices[0]]
                
                if top_confidence > CONFIDENCE_THRESHOLD and has_speech:
                    emotion_placeholder.markdown(f"## 🎭 Emotion: **{top_emotion.capitalize()}**")
                    confidence_placeholder.progress(float(top_confidence))
                    
                    e2_label = model.config.id2label[top_3_indices[1]].capitalize()
                    e2_conf = smoothed_probs[top_3_indices[1]]
                    e3_label = model.config.id2label[top_3_indices[2]].capitalize()
                    e3_conf = smoothed_probs[top_3_indices[2]]

                    top1_placeholder.markdown(f"1. **{top_emotion.capitalize()}** ({top_confidence:.1%})")
                    top2_placeholder.markdown(f"2. {e2_label} ({e2_conf:.1%})")
                    top3_placeholder.markdown(f"3. {e3_label} ({e3_conf:.1%})")
                else:
                    emotion_placeholder.markdown("## 🤔 Emotion: **Listening...**")
                    confidence_placeholder.progress(0.0)
                    top1_placeholder.markdown("1. ...")
                    top2_placeholder.markdown("2. ...")
                    top3_placeholder.markdown("3. ...")

                if len(st.session_state.emotion_history) > 1:
                    emotion_labels = list(model.config.id2label.values())
                    history_df = pd.DataFrame(
                        st.session_state.emotion_history,
                        columns=emotion_labels
                    )
                    if len(history_df) > 60:
                        history_df = history_df.tail(60)
                    chart_placeholder.line_chart(history_df)

        # --- NEW: FINAL TRANSCRIPTION CHECK (Runs when you click STOP) ---
        if len(speech_buffer) > 0:
             with st.spinner("Finalizing transcription..."):
                speech_audio = np.concatenate(speech_buffer)
                speech_buffer = []
                
                input_values = asr_processor(
                    speech_audio, 
                    sampling_rate=SAMPLING_RATE, 
                    return_tensors="pt"
                ).input_values
                
                with torch.no_grad():
                    logits = asr_model(input_values).logits
                
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = asr_processor.batch_decode(predicted_ids)[0]
                
                if len(transcription) > 0:
                    transcription_placeholder.markdown(f"**You said:** *{transcription.lower()}*")
        # ---------------------------------------------------------------

        stream.stop()
        stream.close()
        st.warning("🛑 Listening stopped.")

elif mode == "📂 File":
    uploaded_file = st.file_uploader(
        "Upload an audio file (WAV or MP3)",
        type=["wav", "mp3"]
    )
    
    if uploaded_file is not None:
        with st.spinner(f"Analyzing '{uploaded_file.name}'..."):
            
            try:
                y, sr = librosa.load(uploaded_file, sr=SAMPLING_RATE)
            except Exception as e:
                st.error(f"Error loading audio file: {e}")
                st.stop()
                
            st.info("Applying noise reduction to file...")
            y_clean = nr.reduce_noise(y=y, sr=SAMPLING_RATE)
            st.info("Noise reduction complete. Starting analysis...")
            
            st.session_state.emotion_history = []
            audio_buffer = np.zeros(WINDOW_SAMPLES, dtype='float32')
            smoothed_probs = np.zeros(num_labels, dtype='float32')
            uniform_probs = np.full(num_labels, 1.0 / num_labels, dtype='float32')
            
            buffer_is_warm = False
            warmup_chunks_needed = WINDOW_SAMPLES // SAMPLES_PER_STEP
            current_chunks = 0
            
            st.markdown("Processing file:")
            progress_bar = st.progress(0.0)
            
            total_chunks = len(y) // SAMPLES_PER_STEP
            
            for i in range(total_chunks):
                chunk = y_clean[i * SAMPLES_PER_STEP : (i + 1) * SAMPLES_PER_STEP]
                
                if len(chunk) < SAMPLES_PER_STEP:
                    chunk = np.pad(chunk, (0, SAMPLES_PER_STEP - len(chunk)))
                
                audio_buffer = np.roll(audio_buffer, -len(chunk))
                audio_buffer[-len(chunk):] = chunk
                
                if not buffer_is_warm:
                    current_chunks += 1
                    if current_chunks >= warmup_chunks_needed:
                        buffer_is_warm = True
                    continue

                has_speech = False
                num_vad_chunks = len(chunk) // VAD_CHUNK_SIZE
                for j in range(num_vad_chunks):
                    start = j * VAD_CHUNK_SIZE
                    end = start + VAD_CHUNK_SIZE
                    vad_input_tensor = torch.tensor(chunk[start:end])
                    speech_prob = vad_model(vad_input_tensor, SAMPLING_RATE).item()
                    if speech_prob > 0.5:
                        has_speech = True
                        break

                if has_speech:
                    audio_tensor = torch.tensor(audio_buffer)
                    inputs = processor(audio_tensor, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
                    with torch.no_grad():
                        logits = model(**inputs).logits
                    current_probs = torch.softmax(logits, dim=1)[0].numpy()
                    smoothed_probs = ALPHA * current_probs + (1 - ALPHA) * smoothed_probs
                else:
                    smoothed_probs = 0.05 * uniform_probs + 0.95 * smoothed_probs

                st.session_state.emotion_history.append(smoothed_probs)
                
                top_3_indices = np.argsort(smoothed_probs)[-3:][::-1]
                top_emotion = model.config.id2label[top_3_indices[0]]
                top_confidence = smoothed_probs[top_3_indices[0]]
                
                if top_confidence > CONFIDENCE_THRESHOLD and has_speech:
                    emotion_placeholder.markdown(f"## 🎭 Emotion: **{top_emotion.capitalize()}**")
                    confidence_placeholder.progress(float(top_confidence))
                    
                    e2_label = model.config.id2label[top_3_indices[1]].capitalize()
                    e2_conf = smoothed_probs[top_3_indices[1]]
                    e3_label = model.config.id2label[top_3_indices[2]].capitalize()
                    e3_conf = smoothed_probs[top_3_indices[2]]

                    top1_placeholder.markdown(f"1. **{top_emotion.capitalize()}** ({top_confidence:.1%})")
                    top2_placeholder.markdown(f"2. {e2_label} ({e2_conf:.1%})")
                    top3_placeholder.markdown(f"3. {e3_label} ({e3_conf:.1%})")
                else:
                    emotion_placeholder.markdown("## 🤔 Emotion: **Analyzing...**")
                    confidence_placeholder.progress(0.0)
                    top1_placeholder.markdown("1. ...")
                    top2_placeholder.markdown("2. ...")
                    top3_placeholder.markdown("3. ...")
                    transcription_placeholder.empty()

                    if i % 10 == 0 and len(st.session_state.emotion_history) > 1:
                        emotion_labels = list(model.config.id2label.values())
                        history_df = pd.DataFrame(
                            st.session_state.emotion_history,
                            columns=emotion_labels
                        )
                        if len(history_df) > 60:
                            history_df = history_df.tail(60)
                    
                        chart_placeholder.line_chart(history_df)
                
                progress_bar.progress((i + 1) / total_chunks)

        st.success(f"✅ Analysis complete!")
        st.balloons()
        
        if len(st.session_state.emotion_history) > 1:
            emotion_labels = list(model.config.id2label.values())
            history_df = pd.DataFrame(
                st.session_state.emotion_history,
                columns=emotion_labels
            )
            chart_placeholder.line_chart(history_df)
       

st.markdown("---")
st.markdown("### Session Report")


if len(st.session_state.emotion_history) > 1:
    emotion_labels = list(model.config.id2label.values())
    
    final_history_df = pd.DataFrame(
        st.session_state.emotion_history,
        columns=emotion_labels
    )
    
    final_history_df["Time (s)"] = final_history_df.index * STEP_DURATION
    
    csv_data = final_history_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Session Report (CSV)",
        data=csv_data,
        file_name="emotion_session_report.csv",
        mime="text/csv",
    )
else:
    st.caption("No session data to download. Press 'Start Listening' to record a session.")

if st.button("🧹 Clear History & Chart"):
    st.session_state.emotion_history = []
    chart_placeholder.empty()
    st.rerun()
