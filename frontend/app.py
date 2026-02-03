import streamlit as st
import streamlit.components.v1 as components
import requests
import os
import time
import uuid
import tempfile
import sounddevice as sd
import wavio
import pyperclip
from faster_whisper import WhisperModel

# Page config with light theme and centered layout
st.set_page_config(
    page_title="Multimodal RAG Chatbot",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for minimalistic and engaging design with consistent color scheme
st.markdown("""
<style>

/* ===== Global ===== */
html, body, [class*="css"] {
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background-color: #0E1117;
    color: #E5E7EB;
}

/* ===== Main Area ===== */
.main {
    background-color: #0E1117;
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1F2937;
}

.sidebar-header {
    text-align: center;
    font-size: 20px;
    font-weight: 600;
    color: #E5E7EB;
    margin-bottom: 12px;
}

/* ===== Titles ===== */
h1, h2, h3 {
    color: #E5E7EB;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    color: #9CA3AF;
    font-size: 16px;
    margin-bottom: 30px;
}

/* ===== Buttons ===== */
.stButton button {
    background-color: #3B82F6;
    color: white;
    border-radius: 10px;
    padding: 10px 16px;
    border: none;
    font-weight: 500;
}

.stButton button:hover {
    background-color: #2563EB;
    transform: translateY(-1px);
}

/* ===== File Uploader ===== */
.stFileUploader {
    background-color: #161B22;
    border: 1px dashed #374151;
    border-radius: 12px;
    padding: 14px;
}

/* ===== Selectbox ===== */
div[data-baseweb="select"] {
    background-color: #161B22 !important;
    border-radius: 10px;
    border: 1px solid #374151;
}

/* ===== Chat Messages ===== */
.stChatMessage {
    background-color: #161B22;
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 12px;
    border: 1px solid #1F2937;
}

/* User message */
.stChatMessage[data-testid="user"] {
    border-left: 3px solid #3B82F6;
    background: linear-gradient(135deg, #1b2330 0%, #0f1724 100%);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04), 0 6px 14px rgba(0, 0, 0, 0.45);
}

/* Assistant message */
.stChatMessage[data-testid="assistant"] {
    border-left: 3px solid #22D3EE;
    background: linear-gradient(135deg, #202a36 0%, #131b28 100%);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04), 0 6px 14px rgba(0, 0, 0, 0.45);
}

/* ===== Chat Input ===== */
.stChatInput textarea {
    background-color: #161B22;
    color: #E5E7EB;
    border-radius: 14px;
    border: 1px solid #374151;
    padding: 14px;
}

/* ===== Info / Status ===== */
.status-caption {
    font-size: 13px;
    color: #9CA3AF;
    margin-top: 6px;
}

/* ===== Divider ===== */
hr {
    border: none;
    border-top: 1px solid #1F2937;
    margin: 20px 0;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.clear-chat-btn {
    position: absolute;
    top: 0;
    right: 0;
    margin-top: -50px;
    margin-right: 0;
    z-index: 9999;
}

.clear-chat-btn button {
    background-color: #3B82F6 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.3s ease !important;
}

.clear-chat-btn button:hover {
    background-color: #2563EB !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4) !important;
}

/* Chat history container */
.chat-history-container {
    padding-bottom: 20px;
    margin-bottom: 80px;
}

/* Voice button styling */
div[data-testid="column"]:nth-of-type(2) .stButton button {
    background-color: #3B82F6 !important;
    color: white !important;
    border-radius: 12px !important;
    height: 50px !important;
    font-size: 1.2rem !important;
    border: none !important;
}

div[data-testid="column"]:nth-of-type(2) .stButton button:hover {
    background-color: #2563EB !important;
}
</style>
""", unsafe_allow_html=True)


# Header with engaging title
st.markdown("<h1 style='text-align:center;'>Multimodal RAG CDAC-Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload files and chat across text, image, audio, video & data</p>", unsafe_allow_html=True)



# API endpoints (configurable via env for deployment)
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8001")
API_URL = f"{BACKEND_URL}/chat"
UPLOAD_VIDEO_URL = f"{BACKEND_URL}/upload_video"
UPLOAD_AUDIO_URL = f"{BACKEND_URL}/upload_audio"
UPLOAD_PDF_URL = f"{BACKEND_URL}/upload_pdf"
UPLOAD_IMAGE_URL = f"{BACKEND_URL}/upload_image"
UPLOAD_CSV_URL = f"{BACKEND_URL}/upload_csv"

# ============================
# WHISPER MODEL (CACHED)
# ============================
@st.cache_resource
def load_whisper():
    """Load Whisper model for local speech-to-text conversion"""
    return WhisperModel("base", device="cpu", compute_type="int8")

whisper_model = load_whisper()

# ============================
# VOICE INPUT HELPER FUNCTIONS
# ============================
def record_audio(duration=10, fs=44100):
    """Record audio from microphone and save to temp file"""
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wavio.write(tmp.name, recording, fs, sampwidth=2)
        return tmp.name

def transcribe_audio(audio_path):
    """Transcribe audio file to text using Whisper"""
    segments, _ = whisper_model.transcribe(audio_path, language="en")
    text = " ".join(seg.text for seg in segments).strip()
    os.unlink(audio_path)
    return text

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "has_video" not in st.session_state:
    st.session_state.has_video = False
if "has_audio" not in st.session_state:
    st.session_state.has_audio = False
if "has_pdf" not in st.session_state:
    st.session_state.has_pdf = False
if "has_image" not in st.session_state:
    st.session_state.has_image = False
if "has_sql" not in st.session_state:
    st.session_state.has_sql = False
if "upload_complete" not in st.session_state:
    st.session_state.upload_complete = False
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())
if "voice_input" not in st.session_state:
    st.session_state.voice_input = ""
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""
if "temp_input" not in st.session_state:
    st.session_state.temp_input = ""
if "voice_processing" not in st.session_state:
    st.session_state.voice_processing = False
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "user_message_input" not in st.session_state:
    st.session_state.user_message_input = ""

# ============================

# Sidebar for unified upload with enhanced design
with st.sidebar:
    st.markdown("<div class='sidebar-header'>📂 File Upload Center</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #616161;'>Upload multiple files at once</p>", unsafe_allow_html=True)
    
    # Multiple file uploader with key to reset it
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["mp4", "mkv", "avi", "mov", "mp3", "wav", "m4a", "flac", "pdf", "png", "jpg", "jpeg", "csv"],
        help="Supports: Video, Audio, PDF, Image, CSV",
        accept_multiple_files=True,
        key=st.session_state.uploader_key
    )
    
    if uploaded_files:
        st.info(f"📦 {len(uploaded_files)} file(s) selected")
        
        if st.button("🚀 Upload All Files", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_container = st.container()
            upload_successful = True
            
            # Helper to uniquely identify a file selection
            def _file_token(uf):
                size = getattr(uf, "size", None)
                if size is None:
                    try:
                        size = len(uf.getvalue())
                    except Exception:
                        size = 0
                return (uf.name, size)

            # Only process files that haven't been uploaded before
            new_uploads = [uf for uf in uploaded_files if _file_token(uf) not in st.session_state.processed_files]

            if not new_uploads:
                with status_container:
                    st.info("No new files to process.")
                # Reset uploader so previously selected files disappear
                st.session_state.uploader_key = str(uuid.uuid4())
                time.sleep(1)
                st.rerun()
                
            total_files = len(new_uploads)
            
            for idx, uploaded_file in enumerate(new_uploads):
                file_name = uploaded_file.name
                file_ext = file_name.split('.')[-1].lower()
                
                # Determine file type and endpoint
                if file_ext in ["mp4", "mkv", "avi", "mov"]:
                    file_type = "Video"
                    upload_url = UPLOAD_VIDEO_URL
                    session_key = "has_video"
                    icon = "🎥"
                elif file_ext in ["mp3", "wav", "m4a", "flac"]:
                    file_type = "Audio"
                    upload_url = UPLOAD_AUDIO_URL
                    session_key = "has_audio"
                    icon = "🎵"
                elif file_ext == "pdf":
                    file_type = "PDF"
                    upload_url = UPLOAD_PDF_URL
                    session_key = "has_pdf"
                    icon = "📄"
                elif file_ext in ["png", "jpg", "jpeg"]:
                    file_type = "Image"
                    upload_url = UPLOAD_IMAGE_URL
                    session_key = "has_image"
                    icon = "🖼️"
                elif file_ext == "csv":
                    file_type = "CSV"
                    upload_url = UPLOAD_CSV_URL
                    session_key = "has_sql"
                    icon = "📊"
                else:
                    file_type = "Unknown"
                    upload_url = None
                    session_key = None
                    icon = "❓"
                
                if upload_url:
                    with status_container:
                        status = st.empty()
                        status.info(f"{icon} Processing {file_name}...")
                    
                    try:
                        files = {"file": (file_name, uploaded_file.getvalue())}
                        resp = requests.post(upload_url, files=files)
                        
                        if resp.status_code == 200:
                            data = resp.json()
                            with status_container:
                                st.success(f"✅ {file_type}: {file_name}")
                                if file_type == "CSV" and "table" in data:
                                    st.info(f"   Table: `{data['table']}` | Rows: {data['rows_inserted']}")
                            
                            st.session_state[session_key] = True
                            # Mark this file as processed
                            st.session_state.processed_files.add(_file_token(uploaded_file))
                        else:
                            with status_container:
                                st.error(f"❌ Failed to process {file_name}")
                            upload_successful = False
                    except Exception as e:
                        with status_container:
                            st.error(f"❌ Error uploading {file_name}: {str(e)}")
                        upload_successful = False
                else:
                    with status_container:
                        st.warning(f"⚠️ Unsupported file type: {file_name}")
                    upload_successful = False
                
                # Update progress bar
                progress_bar.progress((idx + 1) / total_files)
            
            with status_container:
                st.success("✅ All files processed!")
            
            # Clear uploaded files if upload was successful
            if upload_successful:
                st.session_state.upload_complete = True
                # Reset uploader by changing its key so previous selections disappear
                st.session_state.uploader_key = str(uuid.uuid4())
                time.sleep(1)
                st.rerun()
    
    st.divider()


    
    # Query mode dropdown with enhanced styling
    st.markdown("<div class='sidebar-header'>🔍 Query Mode Selector</div>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='mode-selector'>", unsafe_allow_html=True)
        # Always show all modes, but indicate which ones are available
        available_modes = ["Normal Query (Default)"]
        
        # Add all context modes with status indicators
        if st.session_state.has_video:
            available_modes.append("🎥 Video Context")
        else:
            available_modes.append("🎥 Video Context (Upload video first)")
        
        if st.session_state.has_audio:
            available_modes.append("🎵 Audio Context")
        else:
            available_modes.append("🎵 Audio Context (Upload audio first)")
        
        if st.session_state.has_pdf:
            available_modes.append("📄 PDF Context")
        else:
            available_modes.append("📄 PDF Context (Upload PDF first)")
        
        if st.session_state.has_image:
            available_modes.append("🖼️ Image Context")
        else:
            available_modes.append("🖼️ Image Context (Upload image first)")
        
        if st.session_state.has_sql:
            available_modes.append("📊 SQL Query")
        else:
            available_modes.append("📊 SQL Query (Upload CSV first)")
        
        query_mode = st.selectbox(
            "Select query mode:",
            available_modes,
            help="Choose what context to use for answering"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Show uploaded files status
    st.markdown("<div class='status-caption'>**Uploaded Files Status:**</div>", unsafe_allow_html=True)
    status_items = []
    if st.session_state.has_video:
        status_items.append("✅ Video")
    if st.session_state.has_audio:
        status_items.append("✅ Audio")
    if st.session_state.has_pdf:
        status_items.append("✅ PDF")
    if st.session_state.has_image:
        status_items.append("✅ Image")
    if st.session_state.has_sql:
        status_items.append("✅ CSV/SQL")
    
    if status_items:
        st.markdown(f"<div class='status-caption'>{' | '.join(status_items)}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-caption'>No files uploaded yet</div>", unsafe_allow_html=True)
    
    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        time.sleep(2)
        st.rerun()

# Display chat history with enhanced messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ============================
# INPUT AREA
# ============================
col_text, col_voice = st.columns([4, 1])

query = None

with col_text:
    text_query = st.chat_input("Type your message or click 🎤 to speak...")

with col_voice:
    if st.button("🎤 Voice", key="voice_btn", help="Click to record voice (7s)"):
        try:
            # Show recording status
            status_placeholder = st.empty()
            status_placeholder.info("🎙️ Recording...")
            
            # Record audio for 7 seconds
            audio_path = record_audio(duration=7)
            
            # Clear recording status and show processing
            status_placeholder.empty()
            
            st.toast("Processing voice", icon="⏳")

            # Transcribe silently (no spinner/processing shown)
            transcribed_text = transcribe_audio(audio_path)
            
            # Copy to clipboard using pyperclip (works on Windows)
            if transcribed_text:
                pyperclip.copy(transcribed_text)
                # Show toast notification for 2 seconds
                st.toast("📋 Text copied to clipboard! Paste it in the query box.", icon="✅")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Use text query if available
if text_query:
    query = text_query

# Process the query
if query:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.write(query)
    
    # Determine which flags to set based on query mode
    # Only activate if file is actually uploaded
    use_video = "🎥 Video Context" == query_mode and st.session_state.has_video
    use_audio = "🎵 Audio Context" == query_mode and st.session_state.has_audio
    use_pdf = "📄 PDF Context" == query_mode and st.session_state.has_pdf
    use_image = "🖼️ Image Context" == query_mode and st.session_state.has_image
    use_sql = "📊 SQL Query" == query_mode and st.session_state.has_sql
    
    # Warn if mode selected but no file uploaded
    if "Upload" in query_mode and "first)" in query_mode:
        st.warning(f"⚠️ {query_mode.split('(')[0].strip()} selected but no file uploaded. Using normal query instead.")
    
    # Call backend API with inline loading indicator
    try:
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = requests.post(
                    API_URL,
                    json={
                        "message": query,
                        "use_video": use_video,
                        "use_audio": use_audio,
                        "use_pdf": use_pdf,
                        "use_image": use_image,
                        "use_sql": use_sql,
                    },
                )
                data = response.json()
                
                if "reply" in data:
                    bot_reply = data["reply"]
                else:
                    bot_reply = f"Error: {data.get('error', 'Unknown error')}"
            st.write(bot_reply)
        
        # Add bot message to history after displaying
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Make sure API is running.")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")