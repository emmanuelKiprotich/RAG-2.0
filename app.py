from __future__ import annotations

import hashlib
import io
import math
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from pypdf import PdfReader
from docx import Document
from pptx import Presentation


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("PERSONAL_RAG_DATA", APP_DIR / "data"))
DB_PATH = DATA_DIR / "library.sqlite3"
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 220

ICON_PATH = APP_DIR / "assets" / "personal-rag-icon.png"
st.set_page_config(page_title="Personal RAG Assistant", page_icon=str(ICON_PATH), layout="wide")


def db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY, file_hash TEXT NOT NULL, filename TEXT NOT NULL,
        category TEXT NOT NULL, location TEXT NOT NULL, content TEXT NOT NULL,
        embedding TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(file_hash)")
    return conn


def setting(name: str) -> str:
    session_key = {"KEROCHEAI_API_KEY": "kerocheai_api_key", "KEROCHEAI_BASE_URL": "kerocheai_base_url"}.get(name)
    if session_key and session_key in st.session_state and st.session_state[session_key]:
        return str(st.session_state[session_key])
    try:
        return str(st.secrets.get(name, os.getenv(name, "")))
    except Exception:
        return os.getenv(name, "")


def api_config() -> tuple[str, str]:
    base_url = setting("KEROCHEAI_BASE_URL").rstrip("/") or "http://127.0.0.1:8787/v1"
    key = setting("KEROCHEAI_API_KEY")
    if not key:
        raise RuntimeError("Create a developer API key in KerocheAI Settings → Developer API, then set KEROCHEAI_API_KEY.")
    return base_url, key


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    base_url, key = api_config()
    url = f"{base_url}/{path.lstrip('/')}"
    try:
        response = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload, timeout=180)
        response.raise_for_status()
    except requests.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f" Response: {exc.response.text[:600]}"
        raise RuntimeError(f"KerocheAI request failed at {url}.{detail} {exc}") from exc
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError("KerocheAI returned a response that was not valid JSON.") from exc


def api_models() -> list[str]:
    base_url, key = api_config()
    try:
        response = requests.get(f"{base_url}/models", headers={"Authorization": f"Bearer {key}"}, timeout=10)
        response.raise_for_status()
        return [item["id"] for item in response.json().get("data", []) if item.get("id")]
    except (requests.RequestException, ValueError, AttributeError):
        return []


def chat(messages: list[dict[str, str]], model: str, temperature: float = 0.2) -> str:
    if not model:
        raise RuntimeError("Choose an installed KerocheAI model in the sidebar.")
    result = api_post("chat/completions", {"model": model, "messages": messages, "temperature": temperature})
    try:
        return result["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("KerocheAI chat response did not contain choices[0].message.content.") from exc


def read_document(name: str, raw: bytes) -> list[tuple[str, str]]:
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(raw))
        return [(f"page {i}", page.extract_text() or "") for i, page in enumerate(reader.pages, 1)]
    if ext == ".docx":
        doc = Document(io.BytesIO(raw))
        return [("document", "\n".join(p.text for p in doc.paragraphs) + "\n" + "\n".join(" | ".join(c.text for c in row.cells) for t in doc.tables for row in t.rows))]
    if ext == ".pptx":
        prs = Presentation(io.BytesIO(raw))
        pages = []
        for i, slide in enumerate(prs.slides, 1):
            texts = [shape.text for shape in slide.shapes if getattr(shape, "has_text_frame", False) and shape.text.strip()]
            pages.append((f"slide {i}", "\n".join(texts)))
        return pages
    if ext in {".txt", ".md"}:
        return [("document", raw.decode("utf-8", errors="replace"))]
    raise ValueError("Supported formats: PDF, DOCX, PPTX, TXT, and Markdown.")


def split_text(text: str) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    result, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + CHUNK_SIZE // 2, end)
            if boundary > start:
                end = boundary
        result.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return result


def ingest(upload, category: str) -> tuple[int, str]:
    raw = upload.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    conn = db()
    if conn.execute("SELECT 1 FROM chunks WHERE file_hash=? LIMIT 1", (digest,)).fetchone():
        return 0, "already indexed"
    pages = read_document(upload.name, raw)
    entries: list[tuple[str, str]] = []
    for location, text in pages:
        entries.extend((location, part) for part in split_text(text))
    if not entries:
        return 0, "no extractable text found"
    conn.executemany("INSERT INTO chunks(file_hash,filename,category,location,content,embedding) VALUES(?,?,?,?,?,?)", [
        (digest, upload.name, category, location, content, "")
        for location, content in entries
    ])
    conn.commit()
    return len(entries), "indexed"


def retrieve(question: str, categories: list[str], limit: int = 7) -> list[dict[str, Any]]:
    conn = db()
    rows = conn.execute("SELECT filename,category,location,content,embedding FROM chunks").fetchall()
    rows = [row for row in rows if not categories or row["category"] in categories]
    if not rows:
        return []
    tokenize = lambda text: re.findall(r"[\w'-]+", text.lower())
    query_terms = set(tokenize(question))
    tokenized = [tokenize(row["content"]) for row in rows]
    doc_freq = Counter(term for doc in tokenized for term in set(doc))
    avg_len = sum(map(len, tokenized)) / max(1, len(tokenized))
    scores = []
    for terms in tokenized:
        counts = Counter(terms)
        score = 0.0
        for term in query_terms:
            freq = counts[term]
            if not freq:
                continue
            inverse = math.log(1 + (len(rows) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            score += inverse * freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * len(terms) / max(1, avg_len)))
        scores.append(score)
    selected = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)[:limit]
    return [{"filename": rows[i]["filename"], "category": rows[i]["category"], "location": rows[i]["location"], "content": rows[i]["content"], "score": scores[i]} for i in selected if scores[i] > 0]


def answer(question: str, sources: list[dict[str, Any]], model: str, extend: bool = False) -> str:
    if not sources:
        return "I couldn't find relevant material in your library. Upload or select the documents that cover this topic, or rephrase your question."
    source_text = "\n\n".join(f"[S{i}] {s['filename']} ({s['location']})\n{s['content']}" for i, s in enumerate(sources, 1))
    policy = "Use only the supplied excerpts. If they do not answer the question, say what is missing. Cite claims using [S1], [S2], etc. Do not invent facts."
    if extend:
        policy += " If general academic knowledge is needed beyond these excerpts, clearly label that portion 'Beyond the uploaded notes' and distinguish it from sourced material."
    return chat([{"role": "system", "content": f"You are a careful personal study assistant. {policy}"},
                 {"role": "user", "content": f"Question: {question}\n\nRetrieved excerpts:\n{source_text}"}], model) or "No answer was generated."


def library_rows():
    return db().execute("SELECT filename, category, COUNT(*) AS pieces, MIN(created_at) AS added FROM chunks GROUP BY file_hash ORDER BY filename").fetchall()


def delete_file(filename: str):
    db().execute("DELETE FROM chunks WHERE filename=?", (filename,)).connection.commit()


title_icon, title_text = st.columns([0.09, 0.91], vertical_alignment="center")
with title_icon:
    st.image(str(ICON_PATH), width=58)
with title_text:
    st.title("Personal RAG Assistant")
    st.caption("Upload your own documents, ask questions, and get answers grounded in cited passages from your library.")

with st.sidebar:
    st.header("Your library")
    st.text_input("KerocheAI API URL", value=setting("KEROCHEAI_BASE_URL") or "http://127.0.0.1:8787/v1", key="kerocheai_base_url", help="Use the API base URL shown in KerocheAI → Settings → Developer API.")
    st.text_input("KerocheAI developer key", type="password", key="kerocheai_api_key", help="Create a key in KerocheAI → Settings → Developer API. This stays in this browser session.")
    try:
        model_names = api_models()
    except RuntimeError:
        model_names = []
    configured_model = setting("KEROCHEAI_CHAT_MODEL")
    if model_names:
        model_choices = model_names + ([configured_model] if configured_model and configured_model not in model_names else [])
        model_index = model_choices.index(configured_model) if configured_model in model_choices else 0
        st.selectbox("KerocheAI model", model_choices, index=model_index, key="selected_model")
    else:
        st.text_input("KerocheAI model name", value=configured_model, key="selected_model")
    uploads = st.file_uploader("Add documents", type=["pdf", "docx", "pptx", "txt", "md"], accept_multiple_files=True)
    category = st.selectbox("Document type", ["Lecture notes", "Course outline", "Past paper", "Other"])
    if st.button("Index uploaded documents", type="primary", disabled=not uploads, use_container_width=True):
        progress = st.progress(0)
        for i, file in enumerate(uploads):
            try:
                count, status = ingest(file, category)
                st.toast(f"{file.name}: {count} sections {status}")
            except Exception as exc:
                st.error(f"{file.name}: {exc}")
            progress.progress((i + 1) / len(uploads))
        st.rerun()
    rows = library_rows()
    st.subheader(f"Indexed documents ({len(rows)})")
    for row in rows:
        left, right = st.columns([5, 1])
        left.caption(f"{row['filename']} · {row['category']} · {row['pieces']} passages")
        if right.button("✕", key=f"del-{row['filename']}", help="Remove document"):
            delete_file(row["filename"])
            st.rerun()
    if not setting("KEROCHEAI_API_KEY"):
        st.warning("Create a developer API key in KerocheAI Settings → Developer API, then enter it above.")

tab_chat, tab_guide, tab_setup = st.tabs(["Ask your library", "Exam revision guide", "Setup"])

with tab_chat:
    cats = sorted({row["category"] for row in rows})
    chosen = st.multiselect("Search in", cats, default=cats, help="Choose document groups to use as evidence.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**[S{i}] {source['filename']} · {source['location']}**\n\n> {source['content']}")
    prompt = st.chat_input("Ask a question about your documents")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                hits = retrieve(prompt, chosen)
                text = answer(prompt, hits, st.session_state.get("selected_model", ""))
                st.markdown(text)
                with st.expander("Sources"):
                    for i, source in enumerate(hits, 1):
                        st.markdown(f"**[S{i}] {source['filename']} · {source['location']}**\n\n> {source['content']}")
                st.session_state.messages.append({"role": "assistant", "content": text, "sources": hits})
            except Exception as exc:
                st.error(str(exc))

with tab_guide:
    st.subheader("Build an exam revision guide")
    st.write("Generate study notes, recurring-question analysis, model answers, and final revision lists from the documents you indexed.")
    scope = st.multiselect("Include these document types", cats, default=cats, key="guide_scope")
    extend = st.checkbox("Allow clearly labelled academic knowledge when the notes are insufficient", value=True)
    if st.button("Generate revision guide", type="primary", disabled=not rows):
        query = "Course key definitions, concepts, principles, advantages, disadvantages, comparisons, lists, and examples. Past paper questions and their years or paper labels. Provide evidence for a comprehensive exam revision guide."
        with st.spinner("Finding relevant passages and drafting your guide…"):
            try:
                passages = retrieve(query, scope, limit=35)
                instructions = "Create a structured exam revision guide with: 1. Short Revision Notes organized by topic/subtopic; 2. Most Frequently Asked Questions ranked by repetition with counts and papers/years when available; 3. Answers to all theory questions found in the supplied past-paper passages; 4. Top Definitions; 5. Top Concepts; 6. Top Likely Exam Questions with answers; 7. a one-page Last-Minute Revision Sheet. Use bullets and tables where useful. Cite source markers [S#]. Never claim a frequency/year unless visible in the excerpts. Clearly flag answers that extend beyond the uploaded material."
                context = "\n\n".join(f"[S{i}] {s['filename']} ({s['location']})\n{s['content']}" for i, s in enumerate(passages, 1))
                st.session_state.guide = chat([
                    {"role": "system", "content": "You are a source-grounded exam revision assistant. Do not invent course material, years, or question repetitions. " + instructions},
                    {"role": "user", "content": f"Additional knowledge allowed: {extend}. Build the guide only from this retrieved evidence.\n\n{context}"}], st.session_state.get("selected_model", "")) or "No guide generated."
                st.session_state.guide_sources = passages
            except Exception as exc:
                st.error(str(exc))
    if st.session_state.get("guide"):
        st.markdown(st.session_state.guide)
        with st.expander("Retrieved source passages"):
            for i, source in enumerate(st.session_state.guide_sources, 1):
                st.markdown(f"**[S{i}] {source['filename']} · {source['location']}**\n\n> {source['content']}")
        st.download_button("Download guide as Markdown", st.session_state.guide, file_name="exam-revision-guide.md", mime="text/markdown")

with tab_setup:
    st.subheader("Run your assistant")
    st.markdown("""
1. Install Python 3.10 or newer.
2. In a terminal opened in this folder, install packages with `pip install -r requirements.txt`.
3. In KerocheAI, open **Settings → Developer API**, create an API key, and copy the API base URL. Enter both in the sidebar. Alternatively, set `KEROCHEAI_API_KEY` and `KEROCHEAI_BASE_URL` as environment variables or in `.streamlit/secrets.toml`.
4. Start the app with `streamlit run app.py`.

Documents and their searchable text are stored locally in `data/library.sqlite3`. Retrieval runs locally; only the selected passages and your question are sent to KerocheAI for answer generation. Scanned PDFs need OCR before their text can be indexed. The app supports PDF, DOCX, PPTX, TXT, and Markdown.
""")
