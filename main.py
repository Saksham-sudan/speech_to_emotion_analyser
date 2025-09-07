import joblib
import sounddevice as sd
import numpy as np
import librosa

model = joblib.load("emotion_model.pkl")

emotions = model.classes_

def record_audio(duration=3, sr=22050):
    print("🎤 Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    print("✅ Recording complete")
    return np.squeeze(audio), sr

def extract_features(audio, sr=22050):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

if __name__ == "__main__":
    audio, sr = record_audio(duration=3)
    features = extract_features(audio, sr)

    prediction = model.predict([features])[0]
    proba = model.predict_proba([features])[0]

    print("\n🎯 Predicted Emotion:", prediction)
    print("📊 Confidence Scores:")
    for emotion, score in zip(emotions, proba):
        print(f"{emotion}: {score:.2f}")