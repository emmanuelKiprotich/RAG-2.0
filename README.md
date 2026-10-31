# Personal RAG Assistant with KerocheAI

A local Streamlit study assistant connected to the KerocheAI model server. Upload study documents, ask source-grounded questions, and draft a revision guide from lecture notes and past papers.

## Features

- Upload PDF, DOCX, PPTX, TXT, and Markdown files.
- Label documents as lecture notes, course outlines, past papers, or other.
- Extract and chunk document text, then retrieve relevant passages locally with BM25 ranking.
- Generate cited answers using a model installed in KerocheAI.
- Draft and download an exam revision guide.
- Store the library in local SQLite. Only retrieved passages and the current prompt are sent to KerocheAI for answer generation.

## Windows setup

1. Make sure KerocheAI is running and you have created a key under **Settings → Developer API**.
2. Double-click `setup.bat` in this folder.
3. If no local settings file exists yet, enter the KerocheAI API URL and key when prompted. Setup creates a Python environment, installs the app packages, adds a desktop shortcut with the app icon, and starts the assistant.
4. Later, open **Personal RAG Assistant** from the desktop shortcut. Keep the KerocheAI server running.

The key is stored in `.streamlit/secrets.toml`, which is excluded from source control. If you already have a local secrets file, setup preserves it. You can also enter the API URL and key in the app sidebar for the current session.

## Run manually

Requires Python 3.10+ and a running KerocheAI server with an installed model.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:KEROCHEAI_API_KEY="your-KerocheAI-developer-key"
streamlit run app.py
```

Create the key in KerocheAI under **Settings → Developer API**. The default API base URL is `http://127.0.0.1:8787/v1`. To connect to another KerocheAI host, set `KEROCHEAI_BASE_URL` to the API base URL displayed in KerocheAI Settings. The app retrieves installed models from KerocheAI; choose one in the sidebar. You can also set `KEROCHEAI_CHAT_MODEL` as the default model.

For persistent settings, put the values in `.streamlit/secrets.toml`:

```toml
KEROCHEAI_API_KEY = "your-KerocheAI-developer-key"
KEROCHEAI_BASE_URL = "http://127.0.0.1:8787/v1"
KEROCHEAI_CHAT_MODEL = "your-installed-model-name"
```

Set `PERSONAL_RAG_DATA` to change where the local SQLite library is stored.

## Notes

- Keep KerocheAI and its local model runtime running while using this app.
- Files and extracted passages stay on this computer. Only the question and the retrieved passages are sent to KerocheAI, which uses its local model runtime.
- Scanned/image-only PDFs require OCR before upload.
- The guide generator can only use passages retrieved for that run. For best results, generate guides for one course or a smaller document group at a time.
- Never commit API keys or the local `data/` folder.
