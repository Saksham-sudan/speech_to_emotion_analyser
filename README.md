# 🎙️ Speech to Tone and Emotion Recognition

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

A powerful real-time web application that analyzes speech to detect emotions and tone while transcribing audio. Built with **Streamlit**, **PyTorch**, and state-of-the-art **Transformers** models.

## 🌟 Features

- **🔴 Real-time Analysis**: Live emotion detection from your microphone.
- **📂 File Upload**: Support for analyzing pre-recorded audio files (WAV, MP3).
- **📝 Live Transcription**: Real-time speech-to-text using OpenAI's Whisper model.
- **📈 Interactive Visualizations**: Dynamic emotion timelines and confidence charts.
- **📊 Session Reports**: detailed analytics and exportable session data.
- **🗣️ VAD Integrated**: Voice Activity Detection to filter out silence.

## 🛠️ Tech Stack

- **Framework**: Streamlit
- **Emotion Model**: Custom DistilHuBERT / Wav2Vec2
- **ASR Model**: Whisper Tiny
- **VAD**: Silero VAD
- **Visualization**: Plotly

## 🚀 Installation

1. **Clone the repository** (or download the source code):
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you encounter issues with `torch`, please visit the [PyTorch website](https://pytorch.org/) for the correct installation command for your system.*

## 💻 Usage

Run the application using Streamlit:

```bash
streamlit run main.py
```

The application will open automatically in your default web browser (usually at `http://localhost:8501`).

### Models Note
The application expects a custom emotion model in the `./distilhubert_hybrid_final` directory. Ensure this directory exists and contains the model files. If using the default Hugging Face model, modification to `main.py` might be required.

## 📂 Project Structure

```
├── main.py                     # Primary Application Entry Point
├── app.py                      # Alternative/Legacy App Version
├── requirements.txt            # Python Dependencies
├── distilhubert_hybrid_final/  # Custom Emotion Model Directory
└── README.md                   # Project Documentation
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📄 License

This project is licensed under the MIT License.
