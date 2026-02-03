# RAG-Chatbot

A small Retrieval-Augmented Generation (RAG) chatbot project with a Python backend and a Streamlit frontend.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Repository Structure](#repository-structure)
- [Environment variables (.env)](#environment-variables-env)
- [Setup & Run](#setup--run)
  - [Backend](#backend)
  - [Frontend](#frontend)
- [Usage](#usage)

## Prerequisites

- Python 3.8 or later
- pip
- Optional but recommended: virtual environment (venv)

## Repository Structure

- `backend/` — Python backend (contains `main.py`)
- `frontend/` — Streamlit frontend (contains `app.py`)
- `README.md` — This file

## Environment variables (.env)

This project expects a `.env` file (or environment variables set in your environment) to supply API keys and model configurations. Do NOT commit your real `.env` file to source control.

Recommended variables (example template):

```env
# Models / Providers
OLLAMA_MODEL=gpt-oss:120b-cloud
OLLAMA_URL=https://ollama.com
GEMINI_MODEL_NAME=gemini-2.5-flash

# API keys (fill these with your real keys)
GROQ_API_KEY=
OLLAMA_API_KEY=
GOOGLE_API_KEY=
GROQ_VISION_API_KEY=
```

Notes:
- Fill the values on the right of `=` with your actual API keys or model names.
- `OLLAMA_URL` defaults to `https://ollama.com` in this template; change only if you run a self-hosted Ollama instance or a different endpoint.
- Keep secrets out of git. Instead, add a `.env.example` (with empty values or placeholders) and add `.env` to `.gitignore`.

Suggested steps to add env files safely:

1. Create a local `.env` (from the repository root):
   ```bash
   cat > .env <<'EOT'
   OLLAMA_MODEL=gpt-oss:120b-cloud
   OLLAMA_URL=https://ollama.com
   GEMINI_MODEL_NAME=gemini-2.5-flash
   GROQ_API_KEY=
   OLLAMA_API_KEY=
   GOOGLE_API_KEY=
   GROQ_VISION_API_KEY=
   EOT
   ```

2. Add `.env` to `.gitignore` (if not already present):
   ```bash
   echo ".env" >> .gitignore
   git add .gitignore
   git commit -m "Ignore local .env files"
   ```

3. Add a `.env.example` to the repo (safe to commit) so contributors know which variables are required:
   ```env
   # .env.example
   OLLAMA_MODEL=gpt-oss:120b-cloud
   OLLAMA_URL=https://ollama.com
   GEMINI_MODEL_NAME=gemini-2.5-flash
   GROQ_API_KEY=
   OLLAMA_API_KEY=
   GOOGLE_API_KEY=
   GROQ_VISION_API_KEY=
   ```

4. In CI / production, set environment variables securely through your platform (GitHub Actions secrets, Docker secrets, host environment, etc.) rather than a committed file.

## Setup & Run

Follow these steps to run the project locally.

### Backend

1. Open a terminal and change into the backend directory:
   ```bash
   cd backend
   ```

2. (Recommended) Create and activate a virtual environment:
   - macOS / Linux:
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

3. Install dependencies (if a requirements file exists):
   ```bash
   pip install -r requirements.txt
   ```

4. Run the backend server:
   ```bash
   python main.py
   ```
   The backend should start and listen on the configured host/port (check `main.py` for defaults).

### Frontend

1. Open a new terminal and change into the frontend directory:
   ```bash
   cd frontend
   ```

2. (Optional) Activate a virtual environment (see backend instructions).

3. Install dependencies (if a requirements file exists):
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   This will open the frontend in your browser (typically at `http://localhost:8501`).

## Usage

- Ensure API keys and model variables are set in your `.env` or environment before running; see the Environment variables (.env) section.
- Start the backend first, then the frontend.
- Use the Streamlit interface to interact with the RAG chatbot. The frontend will communicate with the backend API endpoints defined in `backend/main.py`.# RAG-Voice-Chatbot
