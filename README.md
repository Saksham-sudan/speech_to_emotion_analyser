# 🎙️ Speech to Tone and Emotion Analyzer

**Speech to Tone and Emotion Analyzer** is an AI-powered system designed to decode the non-verbal cues in human speech. Optimized for high-noise environments like call centers, it detects underlying emotions—specifically frustration and anger—regardless of the words being spoken.

Unlike standard models trained on clean studio audio, this project features a **hybrid domain-adaptation engine** that processes low-quality, narrowband telephony audio (8kHz), making it robust for real-world telecommunications and customer support analytics.

---

## 🎯 Project Objectives

* **Automated Quality Assurance:** Replaces manual call listening with 100% automated coverage of customer interactions.
* **Tone Detection:** specifically tuned to detect "Negative Tone" (Anger, Frustration, Urgency) in real-time.
* **Low-Resource Deployment:** Engineered to run efficiently on consumer-grade hardware (e.g., NVIDIA GTX 1650 Ti) using knowledge distillation and quantization techniques.

---

## 🏗️ Architecture & Approach

### 1. The Core Model: DistilHuBERT
We utilize `ntu-spml/distilhubert`, a distilled version of the massive HuBERT model. This architecture retains high-performance speech feature extraction capabilities while being **75% smaller and 73% faster**, enabling low-latency inference on edge devices.

### 2. Hybrid Data Strategy
To address the "Simulation vs. Reality" gap in emotion recognition, we combined two distinct datasets:

* **LEGOv2 (Real-World):** Authentic, unscripted recordings of customers interacting with automated bus schedule systems. This captures genuine frustration and "sighs" often missed by actors.
* **CREMA-D (Augmented):** A diverse, acted dataset used to bolster the model's understanding of extreme emotions. We applied a **Telephony Augmentation Pipeline** (Downsampling to 8kHz → Upsampling to 16kHz + Gaussian Noise) to force the model to learn features robust to telephone line degradation.

### 3. Emotion Mapping
The analyzer classifies audio into three actionable business categories:

| Label | ID | Emotions Covered | Business Implication |
| :--- | :--- | :--- | :--- |
| **NEGATIVE** | `0` | Anger, Frustration, Disgust, Fear | 🚨 **Critical:** Requires Supervisor Intervention |
| **NEUTRAL** | `1` | Calm, Indifferent, Standard Tone | ✅ **Normal:** Standard Interaction |
| **POSITIVE** | `2` | Happiness, Excitement, Gratitude | 🌟 **Excellent:** Potential Testimonial/Up-sell |

---

## 🛠️ Installation

### Prerequisites
* Python 3.10+
* CUDA-enabled GPU (Recommended: 4GB VRAM minimum)

### Setup Steps
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/yourusername/speech-to-tone-analyzer.git](https://github.com/yourusername/speech-to-tone-analyzer.git)
    cd speech-to-tone-analyzer
    ```

2.  **Install Dependencies:**
    ```bash
    pip install torch torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    pip install transformers datasets pandas scikit-learn accelerate numpy
    ```

3.  **Prepare Data:**
    * Download the **LEGOv2** dataset.
    * Download the **CREMA-D** dataset.
    * Update the paths in `main.py` under the `CONFIGURATION` section.

---

## 🚀 Usage

### Training the Model
To start the hybrid training pipeline (which handles data loading, augmentation, and model fine-tuning):

```bash
python main.py
