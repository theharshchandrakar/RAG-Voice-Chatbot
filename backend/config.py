"""
Backend Configuration and Initialization Module
Environment variables, API clients, and database setup
"""

import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import google.generativeai as genai
from groq import Groq
from ollama import Client as OllamaClient
from collections import deque

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, ".env")
load_dotenv(dotenv_path)

# Configuration constants
MAX_SQL_LIMIT = 100
MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "30"))
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DIR = os.path.join(script_dir, "chroma_db")
CACHE_DIR = os.path.join(script_dir, "cache")
MODEL_CACHE_PATH = os.path.join(CACHE_DIR, "embedding_model.pkl")

# Conversation memory: last 10 messages (5 user + 5 assistant)
chat_memory = deque(maxlen=10)


def load_embedding_model():
    """Load embedding model from cache or download if needed."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    if os.path.exists(MODEL_CACHE_PATH):
        try:
            with open(MODEL_CACHE_PATH, 'rb') as f:
                embed_fn = pickle.load(f)
            print("✅ Loaded cached embedding model")
            return embed_fn
        except Exception as e:
            print(f"⚠️  Failed to load cached model: {str(e)}, downloading fresh...")
    
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    try:
        with open(MODEL_CACHE_PATH, 'wb') as f:
            pickle.dump(embed_fn, f)
        print("✅ Downloaded and cached embedding model")
    except Exception as e:
        print(f"⚠️  Could not cache model: {str(e)}, using in-memory")
    
    return embed_fn


def initialize_backend():
    """Initialize all backend components."""
    print("\n🚀 Initializing RAG Chatbot Backend Components...")
    
    # Setup directories
    os.makedirs(CHROMA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Load embedding model
    embed_fn = load_embedding_model()
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collections = {}
    collection_names = {
        "video": "chat_video_context",
        "audio": "chat_audio_context",
        "pdf": "chat_pdf_context",
        "image": "chat_image_context"
    }
    
    for col_type, col_name in collection_names.items():
        try:
            collections[col_type] = chroma_client.get_or_create_collection(
                name=col_name,
                embedding_function=embed_fn
            )
        except Exception as e:
            print(f"⚠️  Could not initialize {col_type} collection: {str(e)}")
    
    # Setup API clients
    groq_client = None
    groq_model = "llama-3.1-8b-instant"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if GROQ_API_KEY:
        try:
            groq_client = Groq(api_key=GROQ_API_KEY)
        except Exception as e:
            print(f"⚠️  Groq initialization failed: {str(e)}")
    
    # Separate Groq Vision client for OCR
    GROQ_VISION_API_KEY = os.getenv("GROQ_VISION_API_KEY")
    groq_vision_client = None
    if GROQ_VISION_API_KEY:
        try:
            groq_vision_client = Groq(api_key=GROQ_VISION_API_KEY)
            print("✅ Groq Vision client initialized for OCR")
        except Exception as e:
            print(f"⚠️  Groq Vision initialization failed: {str(e)}")
    else:
        print("⚠️  GROQ_VISION_API_KEY not set - OCR will be unavailable")
    
    # Setup Gemini
    gemini_model = None
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if GOOGLE_API_KEY:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        except Exception as e:
            print(f"⚠️  Gemini initialization failed: {str(e)}")
    
    # Setup Ollama
    ollama_client = None
    ollama_model = "gpt-oss:120b-cloud"
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
    OLLAMA_URL = os.getenv("OLLAMA_URL", "https://ollama.com")
    if OLLAMA_URL:
        try:
            headers = {'Authorization': f'Bearer {OLLAMA_API_KEY}'} if OLLAMA_API_KEY else None
            ollama_client = OllamaClient(host=OLLAMA_URL, headers=headers)
        except Exception as e:
            print(f"⚠️  Ollama initialization failed: {str(e)}")
    
    # Setup database
    db_path = Path(script_dir) / "sql_data.db"
    
    print("✅ Backend initialization complete!\n")
    
    return {
        "chroma_client": chroma_client,
        "collections": collections,
        "groq_client": groq_client,
        "groq_model": groq_model,
        "groq_vision_client": groq_vision_client,
        "gemini_model": gemini_model,
        "ollama_client": ollama_client,
        "ollama_model": ollama_model,
        "db_path": db_path,
    }


def get_context_window():
    """Get conversation history."""
    return list(chat_memory)


def remember_exchange(user_text: str, assistant_text: str):
    """Store conversation exchange in memory."""
    try:
        chat_memory.append({"role": "user", "content": user_text})
        chat_memory.append({"role": "assistant", "content": assistant_text})
    except Exception:
        pass


def build_messages_with_context(user_message: str, system_prompt: str | None = None):
    """Build message list with conversation context."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for m in get_context_window():
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages
