"""
rag.py — Upgraded RAG system.
Accepts TXT, PDF, and DOCX.
Auto-load on startup is deferred to avoid crashing before API key is available.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

vector_store = None
indexed_docs = []


def _extract_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _extract_pdf(path):
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        return f"[PDF read error: {e}]"


def _extract_docx(path):
    try:
        import docx
        doc = docx.Document(path)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        return f"[DOCX read error: {e}]"


def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    elif ext in (".docx", ".doc"):
        return _extract_docx(path)
    else:
        return _extract_txt(path)


def _chunk_text(text, max_chunk_size=500):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= max_chunk_size:
            chunks.append(para)
        else:
            sentences = para.replace(". ", ".|").split("|")
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) <= max_chunk_size:
                    current += sentence + " "
                else:
                    if current.strip():
                        chunks.append(current.strip())
                    current = sentence + " "
            if current.strip():
                chunks.append(current.strip())
    return [c for c in chunks if len(c) > 20]


def build_vector_store(text, doc_name="notes"):
    global vector_store, indexed_docs
    chunks = _chunk_text(text)
    if not chunks:
        return 0
    try:
        embeddings = OpenAIEmbeddings()
        if vector_store is None:
            vector_store = FAISS.from_texts(chunks, embeddings)
        else:
            vector_store.add_texts(chunks)
        if doc_name not in indexed_docs:
            indexed_docs.append(doc_name)
        return len(chunks)
    except Exception:
        return 0


def build_vector_store_from_file(path):
    text = extract_text(path)
    if not text.strip():
        return 0
    return build_vector_store(text, doc_name=os.path.basename(path))


def retrieve_context(question, k=4):
    global vector_store
    if vector_store is None:
        return ""
    try:
        docs = vector_store.similarity_search(question, k=k)
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception:
        return ""


def get_indexed_docs():
    return indexed_docs.copy()


def clear_vector_store():
    global vector_store, indexed_docs
    vector_store = None
    indexed_docs = []


def load_existing_docs():
    """
    Call this AFTER the API key is confirmed available.
    Loads notes.txt and anything in rag_docs/ into the vector store.
    Called from app.py on startup, not at module import time.
    """
    os.makedirs("rag_docs", exist_ok=True)

    if os.path.exists("notes.txt"):
        try:
            text = _extract_txt("notes.txt")
            if text.strip():
                build_vector_store(text, doc_name="notes.txt")
        except Exception:
            pass

    for fname in os.listdir("rag_docs"):
        fpath = os.path.join("rag_docs", fname)
        if os.path.isfile(fpath) and fname.split(".")[-1].lower() in ("txt", "pdf", "docx"):
            try:
                build_vector_store_from_file(fpath)
            except Exception:
                pass