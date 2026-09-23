# Ai-cybershield
# 🛡️ AI-CyberShield

AI-CyberShield is an AI-powered cybersecurity platform designed to detect
multiple types of digital threats from a single Streamlit application.

## 🚀 Features

### Phase 1

- 🌐 Malicious URL Detection
- 📧 Phishing Email Detection
- 🌐 Network Threat Detection
- 📊 Risk Analysis Dashboard

### Phase 2

- 🧑 Face Deepfake Detection
- 🖼️ Image Deepfake Detection
- 🎥 Video Deepfake Detection
- 🎵 Audio Deepfake Detection
- 🤖 Explainable AI
- 🔐 Unified Threat Risk Score

## 📁 Project Structure

```text
AI-CyberShield/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── url_dataset.csv
│   ├── email_dataset.csv
│   └── network_dataset.csv
│
├── detectors/
│   ├── __init__.py
│   ├── url_detector.py
│   ├── email_detector.py
│   ├── network_detector.py
│   ├── face_detector.py
│   ├── image_detector.py
│   ├── video_detector.py
│   └── audio_detector.py
│
└── models/
    ├── url_model.pkl
    ├── email_model.pkl
    └── network_model.pkl
