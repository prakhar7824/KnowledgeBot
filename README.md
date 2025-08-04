Knowledge Base Chatbot
======================

Upload docs | Ask questions | Get smart answers | With voice or web

What is this?
-------------
A smart chatbot that lets you upload PDFs, DOCX, TXT, or MD files and ask questions about them.
It uses OpenAI’s GPT to answer ONLY from your content. No hallucinations. No fluff.

Main Features:
--------------
- 🧠 Knowledge base with OpenAI embeddings
- 💬 Text + Voice chat support
- 🌐 Web UI using FastAPI
- 🎛️ Voice style and tone customization
- 🗃️ SQLite-powered vector search

Installation
------------

Basic (no voice):
pip install openai faiss-cpu scikit-learn PyPDF2 python-docx fastapi uvicorn websockets python-multipart

Full (with voice):
pip install openai faiss-cpu scikit-learn PyPDF2 python-docx fastapi uvicorn websockets python-multipart speechrecognition pyttsx3

PyAudio Setup (for voice chat):

- Windows:
  pip install pipwin
  pipwin install pyaudio

- Linux:
  sudo apt-get install portaudio19-dev
  pip install pyaudio

- Mac:
  brew install portaudio
  pip install pyaudio

Usage
-----

1. Demo Mode (sample doc):
   python chatbot.py --demo

2. CLI Mode:
   python chatbot.py --cli

3. Text Chat:
   python chatbot.py --api-key YOUR_API_KEY --mode text

4. Voice Chat:
   python chatbot.py --api-key YOUR_API_KEY --mode voice

5. Web Interface:
   python chatbot.py --api-key YOUR_API_KEY --mode web --host 0.0.0.0 --port 8000

Visit: http://localhost:8000

Voice Styles:
-------------
professional, casual, technical, simple

Tones:
------
helpful, neutral, enthusiastic, calm

Example Prompts:
----------------
- What are the working hours?
- How many vacation days do I get?
- What’s the remote work policy?

The bot replies based only on uploaded documents.

Files & Structure:
------------------
- chatbot.py         → All-in-one logic (chatbot, API, CLI, voice)
- DocumentProcessor  → Extracts and splits doc text
- EmbeddingEngine    → Gets OpenAI text embeddings
- VectorDatabase     → Stores chunks + searches using SQLite
- VoiceInterface     → Handles speech I/O
- WebInterface       → FastAPI web app
- CLIInterface       → CLI for uploads/config/chat

Troubleshooting:
----------------
- Voice not working? Check PyAudio installation.
- Getting embedding errors? Verify your API key.
- Web UI not loading? Make sure port 8000 is free.

License:
--------
MIT – use freely, but don’t act like you wrote it 😉
