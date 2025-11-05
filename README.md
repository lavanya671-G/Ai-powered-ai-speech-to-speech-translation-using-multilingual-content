AI-Powered Speech-to-Speech Translation using Multilingual Content

A real-time AI-powered speech-to-speech translation system that converts spoken language from one language to another with multilingual content support.

# Features#
Real-time Speech Recognition: Convert speech to text in multiple languages

AI-Powered Translation: Advanced neural machine translation

Text-to-Speech Synthesis: Generate natural-sounding speech in target language

Multilingual Support: Support for multiple input and output languages

Web Interface: User-friendly web application

Video Processing: Process video files with audio content

Model Training: Custom model training capabilities

Project Structure
live-ai-speech-translator/
├── static/
│   └── uploads/
│       └── videos/          # Sample video files for testing
├── templates/
│   └── index.html          # Web interface
├── Agile Document/         # Project documentation
├── app.py                 # Main Flask application
├── speech_recognizer.py   # Speech recognition module
├── translator.py          # Translation engine
├── text_to_speech.py      # TTS functionality
├── realtime_pipeline.py   # Real-time processing pipeline
├── translation_pipeline.py # Complete translation workflow
├── data_preprocessor.py   # Data preprocessing utilities
├── train_hf_model.py      # Hugging Face model training
├── model_evaluator.py     # Model evaluation scripts
├── ott_integration.py     # OTT platform integration
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation

# Supported Languages#
English

Hindi

Spanish

French

German

Dutch

Portugese

Chinese

Japan

Korean

Italian

Arabic

Russian

# Audio/Video Formats
Input: MP4, WAV, MP3

Output: MP3, WAV

# Models and Algorithms
Speech Recognition: Whisper, Google Speech Recognition

Translation: Transformer-based models

Text-to-Speech: gTTS, pyttsx3

Audio Processing: Librosa, PyAudio

# Sample Data
The project includes sample video files in static/uploads/videos/:

english-1.mp4 - English speech sample

hindi-2.mp4 - Hindi speech sample

# API Endpoints
POST /api/translate - Text translation

POST /api/speech-to-text - Speech recognition

POST /api/text-to-speech - Speech synthesis

POST /api/translate-audio - End-to-end speech translation

# Contributing
Fork the repository

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request
