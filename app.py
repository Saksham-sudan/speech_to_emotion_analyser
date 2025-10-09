import streamlit as st
import sounddevice as sd
import numpy as np
import torch
import time
import queue
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
)


MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
SAMPLING_RATE = 16000
WINDOW_DURATION = 3
STEP_DURATION = 0.5
SAMPLES_PER_STEP = int(STEP_DURATION * SAMPLING_RATE)
WINDOW_SAMPLES = int(WINDOW_DURATION * SAMPLING_RATE)
CONFIDENCE_THRESHOLD = 0.05
ALPHA = 0.2

@st.cache_resource
def load_model():
    try:
        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    except Exception:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model

processor, model = load_model()


st.set_page_config(page_title="🎙️ Real-Time Speech Emotion Analyzer", layout="centered")
st.title("🎙️ Real-Time Speech Emotion Analyzer (Smoothed)")
st.markdown("This app uses a **sliding window** and **EMA smoothing** for stable, real-time emotion detection.")

run_button = st.toggle("🎧 Start / Stop Listening")

emotion_placeholder = st.empty()
confidence_placeholder = st.empty()


data_queue = queue.Queue()


num_labels = len(model.config.id2label)
smoothed_probs = np.zeros(num_labels, dtype='float32')

def audio_callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    data_queue.put(indata.copy())

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

    while run_button:
        try:
            chunk = data_queue.get(timeout=0.1) 
            chunk = chunk.squeeze()

            audio_buffer = np.roll(audio_buffer, -len(chunk))
            audio_buffer[-len(chunk):] = chunk

            audio_tensor = torch.tensor(audio_buffer)
            inputs = processor(audio_tensor, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)

            with torch.no_grad():
                logits = model(**inputs).logits

            
            current_probs = torch.softmax(logits, dim=1)[0].numpy()

      
            smoothed_probs = ALPHA * current_probs + (1 - ALPHA) * smoothed_probs

            
            pred_id = np.argmax(smoothed_probs)
            confidence = smoothed_probs[pred_id]
            emotion = model.config.id2label[pred_id]

            
            if confidence > CONFIDENCE_THRESHOLD:
                emotion_placeholder.markdown(f"## 🎭 Emotion: **{emotion.capitalize()}**")
                confidence_placeholder.progress(float(confidence))
            else:
                emotion_placeholder.markdown("## 🤔 Emotion: **Listening...**")
                confidence_placeholder.progress(float(confidence))

        except queue.Empty:
            
            time.sleep(0.05)
            continue

    stream.stop()
    stream.close()
    st.warning("🛑 Listening stopped.")
