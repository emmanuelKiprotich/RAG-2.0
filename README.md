# Personal RAG Assistant

A local Streamlit study assistant that prefers KerocheAI and automatically falls back to Ollama on this computer if KerocheAI cannot answer. Upload course documents, ask source-grounded questions, and draft revision guides from lecture notes and past papers.

## Features

- Upload PDF, DOCX, PPTX, TXT, and Markdown files.
- Label documents as lecture notes, course outlines, past papers, or other.
- Store extracted passages in a local SQLite library and retrieve them using BM25.
- Generate cited answers and multi-stage exam revision guides.
- Use KerocheAI first, then fall back to a compatible local Ollama model if needed.
- Bind the web app to localhost with upload size and request protection settings.

## Windows setup

1. Have KerocheAI or Ollama running with at least one model installed. KerocheAI is preferred; Ollama can run the app without a KerocheAI key.
2. Double-click `setup.bat` in this folder.
3. If no local settings file exists, enter the KerocheAI API URL and key, or leave the key blank to use Ollama. Setup creates a Python environment, installs dependencies, adds a desktop shortcut, and starts the app.
4. Later, open **Personal RAG Assistant** from the desktop shortcut. Keep the model server running.

Credentials are stored in `.streamlit/secrets.toml`, which is excluded from source control. Existing local settings are preserved. The API key can also be entered in the app sidebar for the current browser session.

## Run manually

Requires Python 3.10+ and either KerocheAI with a developer key or Ollama with a local model installed.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:KEROCHEAI_API_KEY="your-KerocheAI-developer-key"
streamlit run app.py
```

Create a key in KerocheAI under **Settings → Developer API**. The default API base URL is `http://127.0.0.1:8787/v1`; set `KEROCHEAI_BASE_URL` to the URL shown in KerocheAI Settings if it differs. If KerocheAI is missing, down, times out, or rejects a request, the assistant tries Ollama directly at `http://127.0.0.1:11434`. Set `OLLAMA_BASE_URL` if Ollama uses another address. Install a model in Ollama before relying on fallback. Select an installed model in the sidebar, or set `KEROCHEAI_CHAT_MODEL` as the preferred default.

For persistent settings, put values in `.streamlit/secrets.toml`:

```toml
KEROCHEAI_API_KEY = "your-KerocheAI-developer-key"
KEROCHEAI_BASE_URL = "http://127.0.0.1:8787/v1"
KEROCHEAI_CHAT_MODEL = "your-installed-model-name"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
```

Set `PERSONAL_RAG_DATA` to change where the local SQLite library is stored.

## Operational notes

- Ollama fallback uses a local model and does not require a remote AI API account.
- Files and extracted passages stay on this computer. The current question, up to three recent chat turns, and selected passages go to KerocheAI or the configured Ollama server.
- Retrieval uses lexical BM25 matching; it can miss relevant passages when a question uses very different wording from the notes. Try course keywords and document-type filters.
- Each revision-guide run uses up to 24 relevant passages. For large courses, generate separate guides for lecture notes and past papers.
- Scanned/image-only PDFs require OCR before upload.
- Uploads are limited to 50 MB and 5,000 extracted chunks per file.
- The app binds to `127.0.0.1` by default. Change the Streamlit server configuration only if you intentionally need access from another device.
- Never commit API keys, `.streamlit/secrets.toml`, or the local `data/` folder.
