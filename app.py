import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt

# Load your trained model
model = joblib.load("emotion_model.pkl")
emotions = model.classes_

def record_audio(duration=3, sr=22050):
    st.write("🎤 Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    st.write("✅ Recording complete")
    return np.squeeze(audio), sr

def extract_features(audio, sr=22050):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Streamlit UI
st.title("🎶 Real-Time Speech Emotion Analyzer")

st.write("Press the button below to record a 3-second audio clip and analyze its emotion.")

if st.button("Record & Predict"):
    audio, sr = record_audio()
    features = extract_features(audio, sr)

    pred = model.predict([features])[0]
    proba = model.predict_proba([features])[0]

    # Show prediction
    st.subheader(f"🎯 Predicted Emotion: **{pred}**")

    # Show confidence bar chart
    st.bar_chart({emo: p for emo, p in zip(emotions, proba)})

    # Show waveform
    fig, ax = plt.subplots()
    ax.plot(audio)
    ax.set_title("Waveform")
    st.pyplot(fig)

    # Optional: spectrogram
    fig2, ax2 = plt.subplots()
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax2)
    ax2.set_title("Mel Spectrogram")
    fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
    st.pyplot(fig2)