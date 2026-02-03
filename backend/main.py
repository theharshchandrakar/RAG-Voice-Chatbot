from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from groq import Groq

# Import modular components
from config import initialize_backend, get_context_window, remember_exchange, build_messages_with_context
from embeddings import retrieve_context
from llm_clients import send_groq_chat, send_ollama_chat
from video_handler import process_video
from audio_handler import process_audio
from pdf_handler import process_pdf
from image_handler import process_image
from csv_handler import process_csv
from database import (
    get_db_schema, execute_sql, enforce_sql_safety, 
    repair_sql, format_sql_table, generate_sql_with_llm
)

script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, ".env")
load_dotenv()

app = FastAPI()

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize all components once at startup
print("\n" + "="*70)
print("INITIALIZING RAG CHATBOT BACKEND")
print("="*70)

backend_config = initialize_backend()

# Extract components
chroma_client = backend_config["chroma_client"]
collections = backend_config["collections"]
video_collection = collections.get("video")
audio_collection = collections.get("audio")
pdf_collection = collections.get("pdf")
image_collection = collections.get("image")

groq_client = backend_config["groq_client"]
GROQ_MODEL = backend_config["groq_model"]

# Separate Groq Vision client for OCR
GROQ_VISION_API_KEY = os.getenv("GROQ_VISION_API_KEY")
groq_vision_client = None
if GROQ_VISION_API_KEY:
    groq_vision_client = Groq(api_key=GROQ_VISION_API_KEY)
    print("✅ Groq Vision client initialized for OCR")
else:
    print("⚠️  GROQ_VISION_API_KEY not set - OCR will be unavailable")

gemini_model = backend_config["gemini_model"]
ollama_client = backend_config["ollama_client"]
OLLAMA_MODEL = backend_config["ollama_model"]
OLLAMA_HOST = os.getenv("OLLAMA_URL", "https://ollama.com")

DB_PATH = backend_config["db_path"]

print(f"\n✅ Backend ready to accept requests!\n")

# Configuration
MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "30"))


class ChatRequest(BaseModel):
    message: str
    use_video: bool = False
    use_audio: bool = False
    use_pdf: bool = False
    use_image: bool = False
    use_sql: bool = False


@app.get("/")
async def root():
    return {"status": "Chatbot API running"}


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Transcribe video using Groq Whisper and store chunks in ChromaDB"""
    try:
        file_bytes = await file.read()
        result = await process_video(file_bytes, file.filename, groq_client, video_collection)
        return result
    except Exception as e:
        print(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """Analyze image via Gemini, chunk text and store in ChromaDB"""
    try:
        file_bytes = await file.read()
        result = await process_image(file_bytes, file.filename, image_collection, gemini_model)
        return result
    except Exception as e:
        print(f"Image Upload Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """Transcribe audio using Groq Whisper and store chunks in ChromaDB"""
    try:
        file_bytes = await file.read()
        result = await process_audio(file_bytes, file.filename, groq_client, audio_collection)
        return result
    except Exception as e:
        print(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV and create SQL table for querying."""
    try:
        file_bytes = await file.read()
        result = await process_csv(file_bytes, file.filename, str(DB_PATH))
        return result
    except Exception as e:
        print(f"CSV Upload Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Extract text from PDF and store chunks in ChromaDB (with intelligent OCR)"""
    try:
        file_bytes = await file.read()
        result = await process_pdf(
            file_bytes, 
            file.filename, 
            pdf_collection, 
            groq_vision_client, 
            MAX_PDF_SIZE_MB
        )
        return result
    except Exception as e:
        print(f"PDF Upload Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest):
    """Send message with optional video, audio, pdf, image, or SQL context"""
    if not request.message.strip():
        return {"error": "Message cannot be empty"}

    # --- SQL MODE ---
    if request.use_sql:
        try:
            schema = get_db_schema(DB_PATH)
            if "No database found" in schema or "Database is empty" in schema:
                return {
                    "reply": "No SQL database available. Please upload a CSV file first.",
                    "source": "SQL Context"
                }
            
            print(f"SQL Query Request: {request.message}")
            print(f"Schema: {schema[:200]}...")
            
            # Build messages with context for SQL generation
            messages_with_context = build_messages_with_context(request.message)
            
            # Generate SQL WITH prior conversation context for follow-up questions
            raw_sql = generate_sql_with_llm(
                request.message, 
                schema, 
                groq_client,
                GROQ_MODEL,
                ollama_client,
                OLLAMA_MODEL,
                OLLAMA_HOST,
                messages_with_context
            )
            print(f"Generated SQL (raw): {raw_sql}")
            
            # Repair and enforce safety
            fixed_sql = repair_sql(raw_sql)
            safe_sql = enforce_sql_safety(fixed_sql)
            print(f"Final SQL: {safe_sql}")
            
            # Execute query
            rows = execute_sql(safe_sql, DB_PATH)
            table = format_sql_table(rows)
            
            # Format response with proper markdown (clean spacing for tables)
            response_text = (
                f"**SQL Query:**\n" \
                f"```sql\n{safe_sql}\n```\n\n" \
                f"**Results:** ({len(rows)} rows)\n\n" \
                f"{table}"
            )
            
            # Remember SQL exchange in conversation history
            remember_exchange(request.message, response_text)
            
            return {
                "reply": response_text,
                "source": "SQL Query",
                "sql": safe_sql,
                "row_count": len(rows),
                "rows": rows[:10]  # Limit to first 10 for response size
            }
        except Exception as e:
            print(f"SQL Query Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"SQL query failed: {str(e)}",
                "source": "SQL Error"
            }

    # --- VIDEO MODE ---
    if request.use_video:
        context = retrieve_context(request.message, video_collection, n_results=5)

        if not context:
            return {
                "reply": "Question is not in this video (no matching transcript context).",
                "source": "Video Context"
            }

        system_instr = "Answer the question ONLY based on the video transcript provided. If not found, say question is not related to the video."
        user_msg = f"Video Transcript Context:\n{context}\n\nQuestion: {request.message}"

        try:
            # Groq first for video (include conversation context)
            messages = build_messages_with_context(user_msg, system_instr)
            reply = send_groq_chat(groq_client, GROQ_MODEL, messages, temperature=0.2)
            remember_exchange(request.message, reply)
            return {"reply": f"GROQ answer ({GROQ_MODEL}) [Video Mode]: \n{reply}", "source": "Groq (Video)"}
        except Exception:
            # Ollama fallback
            messages = build_messages_with_context(user_msg, system_instr)
            ollama_reply = send_ollama_chat(ollama_client, OLLAMA_MODEL, messages)
            remember_exchange(request.message, ollama_reply)
            return {"reply": ollama_reply, "source": "Ollama (Video Fallback)"}

    # --- AUDIO MODE ---
    if request.use_audio:
        context = retrieve_context(request.message, audio_collection, n_results=5)

        if not context:
            return {
                "reply": "Question is not in this audio (no matching transcript context).",
                "source": "Audio Context"
            }

        system_instr = "Answer the question ONLY based on the audio transcript provided. If not found, say question is not related to the audio."
        user_msg = f"Audio Transcript Context:\n{context}\n\nQuestion: {request.message}"

        try:
            messages = build_messages_with_context(user_msg, system_instr)
            ollama_reply = send_ollama_chat(ollama_client, OLLAMA_MODEL, messages)
            if "Ollama Fallback Error" not in ollama_reply:
                remember_exchange(request.message, ollama_reply)
                return {"reply": ollama_reply, "source": "Ollama (Audio)"}
            
            # Fallback to Groq
            messages = build_messages_with_context(user_msg, system_instr)
            reply = send_groq_chat(groq_client, GROQ_MODEL, messages, temperature=0.2)
            remember_exchange(request.message, reply)
            return {"reply": f"GROQ answer ({GROQ_MODEL}) [Audio Mode]: \n{reply}", "source": "Groq (Audio Fallback)"}
        except Exception as e:
            return {"error": f"Audio query failed: {str(e)}"}

    # --- PDF MODE ---
    if request.use_pdf:
        context = retrieve_context(request.message, pdf_collection, n_results=5)

        if not context:
            return {
                "reply": "Question is not in this PDF (no matching document context).",
                "source": "PDF Context"
            }

        system_instr = "Answer the question ONLY based on the PDF document provided. If not found, say question is not related to the PDF."
        user_msg = f"PDF Context:\n{context}\n\nQuestion: {request.message}"

        try:
            messages = build_messages_with_context(user_msg, system_instr)
            ollama_reply = send_ollama_chat(ollama_client, OLLAMA_MODEL, messages)
            if "Ollama Fallback Error" not in ollama_reply:
                remember_exchange(request.message, ollama_reply)
                return {"reply": ollama_reply, "source": "Ollama (PDF)"}
            
            # Fallback to Groq
            messages = build_messages_with_context(user_msg, system_instr)
            reply = send_groq_chat(groq_client, GROQ_MODEL, messages, temperature=0.2)
            remember_exchange(request.message, reply)
            return {"reply": f"GROQ answer ({GROQ_MODEL}) [PDF Mode]: \n{reply}", "source": "Groq (PDF Fallback)"}
        except Exception as e:
            return {"error": f"PDF query failed: {str(e)}"}

    # --- IMAGE MODE ---
    if request.use_image:
        print(f"\n🖼️  IMAGE MODE QUERY: {request.message}")
        context = retrieve_context(request.message, image_collection, n_results=5)

        if not context or not context.strip():
            print("⚠️  No context retrieved from image collection")
            return {
                "reply": "No image has been uploaded yet, or the question cannot be matched with the uploaded image content. Please upload an image first.",
                "source": "Image Context"
            }

        print(f"✅ Context retrieved ({len(context)} chars)")
        system_instr = "Answer the question based on the image description and extracted text provided below. Be helpful and informative."
        user_msg = f"Image Context:\n{context}\n\nQuestion: {request.message}"

        try:
            # Ollama first
            messages = build_messages_with_context(user_msg, system_instr)
            ollama_reply = send_ollama_chat(ollama_client, OLLAMA_MODEL, messages)
            if "Ollama Fallback Error" not in ollama_reply:
                remember_exchange(request.message, ollama_reply)
                return {"reply": ollama_reply, "source": "Ollama (Image)"}
            # Groq fallback
            messages = build_messages_with_context(user_msg, system_instr)
            reply = send_groq_chat(groq_client, GROQ_MODEL, messages, temperature=0.2)
            remember_exchange(request.message, reply)
            return {"reply": f"GROQ answer ({GROQ_MODEL}) [Image Mode]: \n{reply}", "source": "Groq (Image Fallback)"}
        except Exception as e:
            return {"error": f"Image query failed: {str(e)}"}

    # --- NORMAL MODE: Ollama first, then Groq fallback ---
    try:
        # Build messages with recent context window
        messages = build_messages_with_context(request.message, system_prompt="Use the following recent conversation for context and coherence.")
        ollama_reply = send_ollama_chat(ollama_client, OLLAMA_MODEL, messages)
        
        if "Fallback Error" not in ollama_reply:
            remember_exchange(request.message, ollama_reply)
            return {"reply": ollama_reply, "source": "Ollama"}
        
        print(f"Ollama API Failed. Attempting Groq fallback...")
        reply = send_groq_chat(groq_client, GROQ_MODEL, messages, temperature=0.7, max_tokens=600)
        formatted_reply = f"GROQ answer ({GROQ_MODEL}): {reply}"
        remember_exchange(request.message, formatted_reply)
        return {"reply": formatted_reply, "source": "Groq (Fallback)"}

    except Exception as groq_err:
        return {"error": f"Both AI services failed. Ollama: {ollama_reply} | Groq: {str(groq_err)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)