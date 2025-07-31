#!/usr/bin/env python3
"""
Complete Knowledge Base Voice Chatbot
Single script implementation with all functionalities
"""

import os
import json
import threading
import time
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import speech_recognition
import pyttsx3
import pyaudio

# Core dependencies
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import pickle

# Voice processing
import speech_recognition as sr
import pyttsx3

# Web interface
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# Document processing
import PyPDF2
import docx
from io import BytesIO
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles loading and processing of various document types"""
    
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF bytes"""
        try:
            pdf_file = BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file_content: bytes) -> str:
        """Extract text from DOCX bytes"""
        try:
            doc = docx.Document(BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(file_content: bytes) -> str:
        """Extract text from TXT bytes"""
        try:
            return file_content.decode('utf-8')
        except Exception as e:
            logger.error(f"Error extracting TXT text: {e}")
            return ""
    
    @staticmethod
    def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence end
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - chunk_overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if chunk.strip()]

class EmbeddingEngine:
    """Handles text embeddings using OpenAI API"""
    
    def __init__(self, openai_api_key: str):
        openai.api_key = openai_api_key
        self.model = "text-embedding-ada-002"
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            response = openai.Embedding.create(
                model=self.model,
                input=text
            )
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            if embedding:
                embeddings.append(embedding)
        return embeddings

class VectorDatabase:
    """Simple vector database implementation with SQLite storage"""
    
    def __init__(self, db_path: str = "knowledge_base.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                chunk_text TEXT,
                embedding BLOB
            )
        ''')
        conn.commit()
        conn.close()
    
    def add_documents(self, filename: str, chunks: List[str], embeddings: List[List[float]]):
        """Add document chunks and embeddings to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for chunk, embedding in zip(chunks, embeddings):
            embedding_blob = pickle.dumps(embedding)
            cursor.execute(
                "INSERT INTO documents (filename, chunk_text, embedding) VALUES (?, ?, ?)",
                (filename, chunk, embedding_blob)
            )
        
        conn.commit()
        conn.close()
    
    def search(self, query_embedding: List[float], k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename, chunk_text, embedding FROM documents")
        
        results = []
        query_emb = np.array(query_embedding).reshape(1, -1)
        
        for row in cursor.fetchall():
            doc_id, filename, chunk_text, embedding_blob = row
            doc_embedding = pickle.loads(embedding_blob)
            doc_emb = np.array(doc_embedding).reshape(1, -1)
            
            similarity = cosine_similarity(query_emb, doc_emb)[0][0]
            results.append({
                'id': doc_id,
                'filename': filename,
                'text': chunk_text,
                'similarity': similarity
            })
        
        conn.close()
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:k]
    
    def clear_database(self):
        """Clear all documents from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents")
        conn.commit()
        conn.close()

class VoiceInterface:
    """Handles speech recognition and text-to-speech"""
    
    def __init__(self):
        self.voice_available = False
        self.recognizer = None
        self.microphone = None
        self.tts_engine = None
        
        try:
            import speech_recognition as sr
            import pyttsx3
            
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.tts_engine = pyttsx3.init()
            self.setup_tts()
            self.voice_available = True
            logger.info("Voice interface initialized successfully")
            
        except Exception as e:
            logger.warning(f"Voice interface not available: {e}")
            logger.info("Voice features will be disabled. Install PyAudio for voice support:")
            logger.info("Windows: pip install PyAudio")
            logger.info("Linux: sudo apt-get install portaudio19-dev && pip install PyAudio")
            logger.info("Mac: brew install portaudio && pip install PyAudio")
        
        self.is_listening = False
    
    def setup_tts(self):
        """Configure text-to-speech settings"""
        if not self.voice_available or not self.tts_engine:
            return
            
        try:
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Prefer female voice if available
                female_voice = next((voice for voice in voices if 'female' in voice.name.lower()), voices[0])
                self.tts_engine.setProperty('voice', female_voice.id)
            
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.8)
        except Exception as e:
            logger.error(f"Error setting up TTS: {e}")
            self.voice_available = False
    
    def listen_once(self) -> str:
        """Listen for a single voice input"""
        if not self.voice_available:
            return "Voice input not available. Please install PyAudio: pip install PyAudio"
            
        try:
            import speech_recognition as sr
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                logger.info("Listening...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            text = self.recognizer.recognize_google(audio)
            logger.info(f"Recognized: {text}")
            return text
        except sr.WaitTimeoutError:
            return "Listening timeout - please try again."
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError as e:
            return f"Could not request results; {e}"
        except Exception as e:
            return f"Error during speech recognition: {e}"
    
    def speak(self, text: str):
        """Convert text to speech"""
        if not self.voice_available:
            logger.info(f"TTS not available. Would speak: {text}")
            return
            
        def _speak():
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                logger.error(f"Error in TTS: {e}")
        
        thread = threading.Thread(target=_speak)
        thread.daemon = True
        thread.start()

class KnowledgeBaseChatbot:
    """Main chatbot class with knowledge base integration"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
        self.embedding_engine = EmbeddingEngine(openai_api_key)
        self.vector_db = VectorDatabase()
        self.document_processor = DocumentProcessor()
        self.voice_interface = VoiceInterface()
        
        # Configuration
        self.voice_style = "professional"
        self.tone = "helpful"
        self.model = "gpt-3.5-turbo"
        self.max_tokens = 500
        self.temperature = 0.3
    
    def configure_voice(self, voice_style: str, tone: str):
        """Configure voice style and tone"""
        self.voice_style = voice_style
        self.tone = tone
        logger.info(f"Voice configured: {voice_style}, {tone}")
    
    def add_document(self, filename: str, file_content: bytes) -> Dict[str, Any]:
        """Add a document to the knowledge base"""
        try:
            # Extract text based on file type
            file_ext = Path(filename).suffix.lower()
            
            if file_ext == '.pdf':
                text = self.document_processor.extract_text_from_pdf(file_content)
            elif file_ext == '.docx':
                text = self.document_processor.extract_text_from_docx(file_content)
            elif file_ext in ['.txt', '.md']:
                text = self.document_processor.extract_text_from_txt(file_content)
            else:
                return {"success": False, "message": f"Unsupported file type: {file_ext}"}
            
            if not text.strip():
                return {"success": False, "message": "No text could be extracted from the document"}
            
            # Split text into chunks
            chunks = self.document_processor.split_text(text)
            
            # Generate embeddings
            embeddings = self.embedding_engine.get_embeddings(chunks)
            
            if not embeddings:
                return {"success": False, "message": "Failed to generate embeddings"}
            
            # Store in vector database
            self.vector_db.add_documents(filename, chunks, embeddings)
            
            return {
                "success": True,
                "message": f"Successfully added {filename} with {len(chunks)} chunks",
                "chunks": len(chunks)
            }
        
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return {"success": False, "message": f"Error processing document: {str(e)}"}
    
    def search_knowledge_base(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search the knowledge base for relevant information"""
        try:
            query_embedding = self.embedding_engine.get_embedding(query)
            if not query_embedding:
                return []
            
            results = self.vector_db.search(query_embedding, k)
            return results
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []
    
    def generate_response(self, user_query: str) -> str:
        """Generate response based on knowledge base"""
        try:
            # Search for relevant context
            relevant_docs = self.search_knowledge_base(user_query)
            
            if not relevant_docs:
                return "I don't have information about that in my knowledge base. Please upload relevant documents first."
            
            # Build context from retrieved documents
            context_parts = []
            for doc in relevant_docs:
                context_parts.append(f"From {doc['filename']}:\n{doc['text']}")
            
            context = "\n\n".join(context_parts)
            
            # Create controlled system prompt
            system_prompt = f"""You are a helpful assistant. ONLY answer based on the provided knowledge base context below.
If information is missing, say "I don't know based on the knowledge provided."
Voice style: {self.voice_style}
Tone: {self.tone}

Context documents:
{context}

Important: Answer only based on the provided context. Do not use external knowledge or make assumptions beyond what's explicitly stated in the context."""

            # Generate response using OpenAI
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Sorry, I encountered an error while processing your request: {str(e)}"
    
    def chat_with_voice(self):
        """Start voice-based chat session"""
        if not self.voice_interface.voice_available:
            print("❌ Voice interface not available!")
            print("📝 Install PyAudio for voice support:")
            print("   Windows: pip install PyAudio")
            print("   Linux: sudo apt-get install portaudio19-dev && pip install PyAudio")
            print("   Mac: brew install portaudio && pip install PyAudio")
            print("\n💬 Switching to text chat mode...")
            self.chat_with_text()
            return
            
        print("Voice chat started! Say 'quit' or 'exit' to stop.")
        self.voice_interface.speak("Hello! I'm ready to answer questions about your uploaded documents.")
        
        while True:
            try:
                user_input = self.voice_interface.listen_once()

                if user_input.lower() in ['quit', 'exit', 'stop', 'goodbye']:
                    self.voice_interface.speak("Goodbye!")
                    break

                if "sorry" in user_input.lower() or "error" in user_input.lower():
                    continue

                print(f"You: {user_input}")
                response = self.generate_response(user_input)
                print(f"Bot: {response}")
                self.voice_interface.speak(response)  # <-- Ensure this line is present

            except KeyboardInterrupt:
                self.voice_interface.speak("Chat ended.")
                break
            except Exception as e:
                logger.error(f"Error in voice chat: {e}")
                continue
    
    def chat_with_text(self):
        """Start text-based chat session"""
        print("Text chat started! Type 'quit' or 'exit' to stop.")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'stop']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                response = self.generate_response(user_input)
                print(f"Bot: {response}")
                self.voice_interface.speak(response)
                
            except KeyboardInterrupt:
                print("\nChat ended.")
                break
            except Exception as e:
                logger.error(f"Error in text chat: {e}")
                continue

# Web Interface using FastAPI
class WebInterface:
    """Web interface for the chatbot"""
    
    def __init__(self, chatbot: KnowledgeBaseChatbot):
        self.chatbot = chatbot
        self.app = FastAPI(title="Knowledge Base Chatbot")
        self.setup_routes()
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_homepage():
            return self.get_html_interface()
        
        @self.app.post("/upload")
        async def upload_document(file: UploadFile = File(...)):
            try:
                content = await file.read()
                result = self.chatbot.add_document(file.filename, content)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/configure")
        async def configure_voice(config: dict):
            try:
                voice_style = config.get("voice_style", "professional")
                tone = config.get("tone", "helpful")
                self.chatbot.configure_voice(voice_style, tone)
                return {"success": True, "message": "Configuration updated"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/chat")
        async def chat_endpoint(message: dict):
            try:
                user_query = message.get("message", "")
                response = self.chatbot.generate_response(user_query)
                return {"response": response}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    
                    if message_data.get("type") == "voice_listen":
                        # Handle voice input
                        if self.chatbot.voice_interface.voice_available:
                            voice_input = self.chatbot.voice_interface.listen_once()
                            await websocket.send_text(json.dumps({
                                "type": "voice_input",
                                "text": voice_input
                            }))
                        else:
                            await websocket.send_text(json.dumps({
                                "type": "voice_error",
                                "message": "Voice input not available. Install PyAudio."
                            }))
                    
                    elif message_data.get("type") == "chat":
                        # Handle chat message
                        user_message = message_data.get("message", "")
                        response = self.chatbot.generate_response(user_message)
                        
                        await websocket.send_text(json.dumps({
                            "type": "chat_response",
                            "response": response
                        }))
                        
                        # Speak the response if voice is enabled and available
                        if (message_data.get("voice_enabled", False) and 
                            self.chatbot.voice_interface.voice_available):
                            self.chatbot.voice_interface.speak(response)
            
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
    
    def get_html_interface(self) -> str:
        """Return HTML interface"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Base Chatbot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 90%;
            max-width: 800px;
            height: 90vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .config-panel {
            padding: 15px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }
        .upload-area {
            padding: 20px;
            border: 2px dashed #4facfe;
            margin: 15px;
            border-radius: 10px;
            text-align: center;
            background: #f8f9ff;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .upload-area:hover {
            background: #e6f3ff;
            border-color: #2196F3;
        }
        .chat-area {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #fafafa;
        }
        .message {
            margin: 10px 0;
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .user-message {
            background: #4facfe;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        .bot-message {
            background: white;
            color: #333;
            border: 1px solid #e9ecef;
        }
        .input-area {
            padding: 20px;
            background: white;
            border-top: 1px solid #e9ecef;
            display: flex;
            gap: 10px;
        }
        input, select, button {
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 25px;
            font-size: 14px;
        }
        input[type="text"] {
            flex: 1;
        }
        button {
            background: #4facfe;
            color: white;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        button:hover {
            background: #2196F3;
            transform: translateY(-2px);
        }
        .voice-btn {
            background: #28a745;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .voice-btn.listening {
            background: #dc3545;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        .status {
            padding: 10px;
            text-align: center;
            font-weight: bold;
        }
        .success { color: #28a745; }
        .error { color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Knowledge Base Chatbot</h1>
            <p>Upload documents and chat with AI about their content</p>
        </div>
        
        <div class="config-panel">
            <select id="voiceStyle">
                <option value="professional">Professional</option>
                <option value="casual">Casual</option>
                <option value="technical">Technical</option>
                <option value="simple">Simple</option>
            </select>
            
            <select id="tone">
                <option value="helpful">Helpful</option>
                <option value="neutral">Neutral</option>
                <option value="enthusiastic">Enthusiastic</option>
                <option value="calm">Calm</option>
            </select>
            
            <button onclick="updateConfig()">Update Config</button>
            
            <label>
                <input type="checkbox" id="voiceEnabled"> Enable Voice
            </label>
        </div>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <input type="file" id="fileInput" style="display: none;" accept=".pdf,.docx,.txt,.md" onchange="uploadFile()">
            <p>📁 Click to upload documents (PDF, DOCX, TXT, MD)</p>
        </div>
        
        <div id="status" class="status"></div>
        
        <div class="chat-area" id="chatArea">
            <div class="message bot-message">
                Hello! Upload some documents and I'll answer questions about them.
            </div>
        </div>
        
        <div class="input-area">
            <input type="text" id="messageInput" placeholder="Ask me anything about your documents..." onkeypress="handleKeyPress(event)">
            <button onclick="sendMessage()">Send</button>
            <button class="voice-btn" id="voiceBtn" onclick="toggleVoiceInput()">🎤</button>
        </div>
    </div>

    <script>
        let ws = null;
        let isListening = false;

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === 'voice_input') {
                    document.getElementById('messageInput').value = data.text;
                    if (data.text && !data.text.includes('Sorry') && !data.text.includes('error')) {
                        sendMessage();
                    }
                    stopVoiceInput();
                } else if (data.type === 'chat_response') {
                    addMessage(data.response, 'bot');
                }
            };
            
            ws.onclose = function() {
                setTimeout(connectWebSocket, 3000);
            };
        }

        function showStatus(message, isError = false) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = `status ${isError ? 'error' : 'success'}`;
            setTimeout(() => status.textContent = '', 3000);
        }

        async function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                showStatus('Uploading and processing document...');
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showStatus(`✅ ${result.message}`);
                    addMessage(`Document "${file.name}" uploaded successfully with ${result.chunks} chunks.`, 'bot');
                } else {
                    showStatus(`❌ ${result.message}`, true);
                }
            } catch (error) {
                showStatus(`❌ Upload failed: ${error.message}`, true);
            }
            
            fileInput.value = '';
        }

        async function updateConfig() {
            const voiceStyle = document.getElementById('voiceStyle').value;
            const tone = document.getElementById('tone').value;
            
            try {
                const response = await fetch('/configure', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ voice_style: voiceStyle, tone: tone })
                });
                
                const result = await response.json();
                showStatus('✅ Configuration updated');
            } catch (error) {
                showStatus('❌ Configuration update failed', true);
            }
        }

        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            addMessage(message, 'user');
            input.value = '';
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'chat',
                    message: message,
                    voice_enabled: document.getElementById('voiceEnabled').checked
                }));
            } else {
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: message })
                    });
                    
                    const result = await response.json();
                    addMessage(result.response, 'bot');
                } catch (error) {
                    addMessage('Sorry, there was an error processing your message.', 'bot');
                }
            }
        }

        function addMessage(text, sender) {
            const chatArea = document.getElementById('chatArea');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            messageDiv.textContent = text;
            
            chatArea.appendChild(messageDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        function toggleVoiceInput() {
            if (isListening) {
                stopVoiceInput();
            } else {
                startVoiceInput();
            }
        }

        function startVoiceInput() {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                showStatus('❌ WebSocket not connected', true);
                return;
            }
            
            isListening = true;
            const voiceBtn = document.getElementById('voiceBtn');
            voiceBtn.classList.add('listening');
            voiceBtn.textContent = '🛑';
            
            showStatus('🎤 Listening...');
            
            ws.send(JSON.stringify({ type: 'voice_listen' }));
        }

        function stopVoiceInput() {
            isListening = false;
            const voiceBtn = document.getElementById('voiceBtn');
            voiceBtn.classList.remove('listening');
            voiceBtn.textContent = '🎤';
            
            document.getElementById('status').textContent = '';
        }

        // Initialize
        connectWebSocket();
    </script>
</body>
</html>
        """

def main():
    """Main function to run the chatbot"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Knowledge Base Voice Chatbot")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--mode", choices=["text", "voice", "web"], default="web", help="Chat mode")
    parser.add_argument("--host", default="0.0.0.0", help="Web host")
    parser.add_argument("--port", type=int, default=8000, help="Web port")
    
    args = parser.parse_args()
    
    # Initialize chatbot
    print("🤖 Initializing Knowledge Base Chatbot...")
    chatbot = KnowledgeBaseChatbot(args.api_key)
    
    if args.mode == "text":
        print("Starting text chat mode...")
        chatbot.chat_with_text()
    
    elif args.mode == "voice":
        print("Starting voice chat mode...")
        chatbot.chat_with_voice()
    
    elif args.mode == "web":
        print(f"Starting web interface on http://{args.host}:{args.port}")
        web_interface = WebInterface(chatbot)
        uvicorn.run(web_interface.app, host=args.host, port=args.port)

def demo_mode():
    """Demo mode with sample documents"""
    print("🚀 Demo Mode - Knowledge Base Chatbot")
    print("=" * 50)
    
    # Get API key
    api_key = input("Enter your OpenAI API key: ").strip()
    if not api_key:
        print("❌ API key is required!")
        return
    
    chatbot = KnowledgeBaseChatbot(api_key)
    
    # Create sample document
    sample_doc = """
    Company Policy Manual
    
    Working Hours:
    Our standard working hours are Monday to Friday, 9:00 AM to 5:00 PM.
    Employees are expected to take a one-hour lunch break between 12:00 PM and 2:00 PM.
    
    Remote Work Policy:
    Employees may work remotely up to 2 days per week with manager approval.
    Remote workers must be available during core hours (10:00 AM to 3:00 PM).
    
    Vacation Policy:
    All full-time employees receive 15 days of paid vacation per year.
    Vacation requests must be submitted at least 2 weeks in advance.
    A maximum of 5 vacation days can be carried over to the next year.
    
    Health Benefits:
    The company provides comprehensive health insurance covering medical and dental.
    Employees contribute 20% of the premium cost.
    Coverage begins on the first day of the month following 30 days of employment.
    """
    
    print("\n📄 Adding sample company policy document...")
    result = chatbot.add_document("company_policy.txt", sample_doc.encode('utf-8'))
    
    if result["success"]:
        print(f"✅ {result['message']}")
    else:
        print(f"❌ {result['message']}")
        return
    
    # Configure chatbot
    chatbot.configure_voice("professional", "helpful")
    print("🎯 Configured for professional, helpful responses")
    
    print("\n" + "=" * 50)
    print("💬 Chat with the AI about the company policy!")
    print("Try asking questions like:")
    print("- What are the working hours?")
    print("- How many vacation days do I get?")
    print("- What is the remote work policy?")
    print("- When does health coverage start?")
    print("\nType 'quit' to exit, 'voice' for voice mode")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'stop']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'voice':
                print("🎤 Switching to voice mode...")
                chatbot.chat_with_voice()
                print("💬 Back to text mode...")
                continue
            
            if not user_input:
                continue
            
            print("🤖 Bot:", end=" ")
            response = chatbot.generate_response(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

class CLIInterface:
    """Command Line Interface for easy interaction"""
    
    def __init__(self, chatbot: KnowledgeBaseChatbot):
        self.chatbot = chatbot
    
    def run(self):
        """Run CLI interface"""
        print("🤖 Knowledge Base Chatbot CLI")
        print("=" * 40)
        print("Commands:")
        print("  upload <filepath>  - Upload a document")
        print("  config <style> <tone> - Configure voice")
        print("  chat              - Start text chat")
        print("  voice             - Start voice chat")
        print("  clear             - Clear knowledge base")
        print("  status            - Show current status")
        print("  help              - Show this help")
        print("  quit              - Exit")
        print("=" * 40)
        
        while True:
            try:
                command = input("\n> ").strip().split()
                
                if not command:
                    continue
                
                cmd = command[0].lower()
                
                if cmd == 'quit':
                    break
                
                elif cmd == 'help':
                    self.show_help()
                
                elif cmd == 'upload':
                    if len(command) < 2:
                        print("❌ Usage: upload <filepath>")
                        continue
                    self.upload_file(command[1])
                
                elif cmd == 'config':
                    if len(command) < 3:
                        print("❌ Usage: config <style> <tone>")
                        continue
                    self.configure_voice(command[1], command[2])
                
                elif cmd == 'chat':
                    self.chatbot.chat_with_text()
                
                elif cmd == 'voice':
                    self.chatbot.chat_with_voice()
                
                elif cmd == 'clear':
                    self.clear_knowledge_base()
                
                elif cmd == 'status':
                    self.show_status()
                
                else:
                    print(f"❌ Unknown command: {cmd}")
                    print("Type 'help' for available commands")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("👋 Goodbye!")
    
    def show_help(self):
        """Show help information"""
        print("\n📚 Available Commands:")
        print("  upload <filepath>     - Upload a document to knowledge base")
        print("  config <style> <tone> - Configure AI voice and tone")
        print("  chat                  - Start interactive text chat")
        print("  voice                 - Start interactive voice chat")
        print("  clear                 - Clear all documents from knowledge base")
        print("  status                - Show current configuration and stats")
        print("  help                  - Show this help message")
        print("  quit                  - Exit the application")
        print("\n🎨 Voice Styles: professional, casual, technical, simple")
        print("🎭 Tones: helpful, neutral, enthusiastic, calm")
    
    def upload_file(self, filepath):
        """Upload a file to knowledge base"""
        try:
            if not os.path.exists(filepath):
                print(f"❌ File not found: {filepath}")
                return
            
            with open(filepath, 'rb') as f:
                content = f.read()
            
            filename = os.path.basename(filepath)
            result = self.chatbot.add_document(filename, content)
            
            if result["success"]:
                print(f"✅ {result['message']}")
            else:
                print(f"❌ {result['message']}")
        
        except Exception as e:
            print(f"❌ Error uploading file: {e}")
    
    def configure_voice(self, style, tone):
        """Configure voice style and tone"""
        valid_styles = ["professional", "casual", "technical", "simple"]
        valid_tones = ["helpful", "neutral", "enthusiastic", "calm"]
        
        if style not in valid_styles:
            print(f"❌ Invalid style. Choose from: {', '.join(valid_styles)}")
            return
        
        if tone not in valid_tones:
            print(f"❌ Invalid tone. Choose from: {', '.join(valid_tones)}")
            return
        
        self.chatbot.configure_voice(style, tone)
        print(f"✅ Voice configured: {style}, {tone}")
    
    def clear_knowledge_base(self):
        """Clear the knowledge base"""
        confirm = input("❓ Are you sure you want to clear all documents? (y/N): ")
        if confirm.lower() == 'y':
            self.chatbot.vector_db.clear_database()
            print("✅ Knowledge base cleared")
        else:
            print("❌ Operation cancelled")
    
    def show_status(self):
        """Show current status"""
        print(f"\n📊 Current Status:")
        print(f"  Voice Style: {self.chatbot.voice_style}")
        print(f"  Tone: {self.chatbot.tone}")
        print(f"  Model: {self.chatbot.model}")
        print(f"  Max Tokens: {self.chatbot.max_tokens}")
        print(f"  Temperature: {self.chatbot.temperature}")
        
        # Count documents in database
        try:
            conn = sqlite3.connect(self.chatbot.vector_db.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT filename) FROM documents")
            doc_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM documents")
            chunk_count = cursor.fetchone()[0]
            conn.close()
            
            print(f"  Documents: {doc_count}")
            print(f"  Text Chunks: {chunk_count}")
        except Exception as e:
            print(f"  Database Status: Error - {e}")

if __name__ == "__main__":
    import sys
    
    # Check if running in demo mode
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == "--demo"):
        demo_mode()
    
    elif len(sys.argv) == 2 and sys.argv[1] == "--cli":
        api_key = input("Enter your OpenAI API key: ").strip()
        if not api_key:
            print("❌ API key is required!")
            sys.exit(1)
        
        chatbot = KnowledgeBaseChatbot(api_key)
        cli = CLIInterface(chatbot)
        cli.run()
    
    else:
        main()

"""
INSTALLATION REQUIREMENTS:

Basic Installation (Text-only mode):
pip install openai faiss-cpu scikit-learn PyPDF2 python-docx fastapi uvicorn websockets python-multipart

Full Installation (with Voice support):
pip install openai faiss-cpu scikit-learn PyPDF2 python-docx fastapi uvicorn websockets python-multipart speechrecognition pyttsx3

For Voice Support (PyAudio):
- Windows: pip install PyAudio
- Linux: sudo apt-get install portaudio19-dev && pip install PyAudio  
- Mac: brew install portaudio && pip install PyAudio

Alternative for Windows PyAudio issues:
pip install pipwin
pipwin install pyaudio

USAGE EXAMPLES:

1. Demo Mode (easiest - works without voice):
   python chatbot.py --demo

2. CLI Mode:
   python chatbot.py --cli

3. Text Chat Mode:
   python chatbot.py --api-key YOUR_API_KEY --mode text

4. Voice Chat Mode (requires PyAudio):
   python chatbot.py --api-key YOUR_API_KEY --mode voice

5. Web Interface Mode:
   python chatbot.py --api-key YOUR_API_KEY --mode web --host 0.0.0.0 --port 8000

FEATURES:
✅ Single script with all functionality
✅ Works without voice (graceful fallback to text)
✅ Multiple document formats (PDF, DOCX, TXT, MD)
✅ Vector database with SQLite storage
✅ OpenAI embeddings and chat completion
✅ Optional voice input/output with speech recognition
✅ Web interface with file upload
✅ Configurable voice style and tone
✅ Strict knowledge base only responses
✅ CLI interface for easy management
✅ Demo mode with sample data
✅ WebSocket support for real-time chat
✅ Error handling and logging
✅ Professional web UI with animations
✅ Graceful degradation when voice not available
"""