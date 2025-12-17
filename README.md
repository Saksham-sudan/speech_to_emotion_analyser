Markdown# Speech to tone and emototion analyzer

**Speech to tone and emototion analyzer** is a lightweight, high-performance Speech Emotion Recognition (SER) system designed specifically for the **Call Center** industry. It automates Quality Assurance (QA) by detecting customer frustration and anger in real-time, enabling supervisors to intervene before churn occurs.

Unlike generic emotion models trained on studio-quality audio, EmoVoice is engineered to handle **noisy, narrowband telephony audio (8kHz)** by leveraging a hybrid training strategy of real-world call data and acoustically augmented studio recordings.

---

## 🎯 Project Objectives

* **Automate QA:** Replace manual call sampling with 100% automated coverage.
* **Domain Adaptation:** specialized for the low-fidelity acoustic environment of telephone networks.
* **Resource Efficiency:** Optimized to run on consumer-grade hardware (e.g., NVIDIA GTX 1650 Ti) using DistilHuBERT and FP16 inference.

---

## 🏗️ Architecture & Approach

### 1. The Model: DistilHuBERT
We utilize `ntu-spml/distilhubert`, a knowledge-distilled version of HuBERT. It retains most of the performance of large speech models while being **75% smaller and 73% faster**, making it ideal for real-time deployment on edge devices or modest servers.

### 2. The Hybrid Dataset Strategy
To solve the "Data Scarcity" problem in SER, we combined two distinct datasets:

* **LEGOv2 (Real World):** Contains authentic, unscripted interactions of customers talking to a bus schedule system. Provides genuine "Frustration" and "Neutral" samples.
* **CREMA-D (Augmented):** A large-scale acted dataset. We applied a **Telephony Augmentation Pipeline** (Downsampling to 8kHz → Upsampling to 16kHz → Adding Line Noise) to force the model to learn features robust to phone quality.

### 3. Class Mapping
The system maps complex emotions into actionable Business KPIs:
* **NEGATIVE (0):** Anger, Frustration, Disgust, Fear (⚠️ Alert Supervisor)
* **NEUTRAL (1):** Standard interaction
* **POSITIVE (2):** Happiness, Excitement

---

## 🛠️ Installation

### Prerequisites
* Python 3.10+
* CUDA-enabled GPU (Recommended: 4GB VRAM minimum)

### Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/emovoice-analytics.git](https://github.com/yourusername/emovoice-analytics.git)
   cd emovoice-analytics
Install dependencies:Bashpip install torch torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install transformers datasets pandas scikit-learn accelerate
🚀 Usage1. Training the ModelThe training script automatically handles data loading, augmentation, and GPU optimization (Gradient Accumulation).Bashpython main.py
Note: Ensure you have downloaded the LEGOv2 and CREMA-D datasets and updated the paths in the CONFIGURATION section of main.py.2. Inference (Prediction)To predict emotions on a new audio file:Pythonfrom transformers import pipeline

classifier = pipeline("audio-classification", model="./distilhubert_hybrid_final")
prediction = classifier("path/to/customer_call.wav")

print(prediction)
# Output: [{'label': 'NEGATIVE', 'score': 0.98}, ...]
📊 Technical Challenges & SolutionsChallengeSolutionCUDA OOM ErrorsImplemented Gradient Accumulation (Steps=16) to simulate large batches on a 4GB GPU.Dataset IncompatibilityBuilt a custom Tokenizer-Level Type Casting pipeline to force all labels to Int64, resolving Windows/PyTorch type conflicts.Domain MismatchApplied On-the-fly Spectral Augmentation to studio data to mimic the frequency response of telephone lines.🔮 Future ScopeSpeaker Diarization: Separate "Agent" vs. "Customer" audio tracks automatically.Dashboard Integration: Build a Streamlit frontend for live visualization of call sentiment.Transcrition (ASR): Integrate OpenAI Whisper to correlate text sentiment with audio emotion.📜 LicenseThis project is open-source and available under the MIT License.
