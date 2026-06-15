"""
RAG pipeline using ChromaDB with per-session collection isolation.

Each session (thread_id) gets its own named collection inside a shared
ChromaDB instance. This means:
  - Multiple users never overwrite each other's documents
  - A session can hold multiple uploaded reference docs
  - Collections persist across app restarts until explicitly deleted
"""

import os
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "uploads", "chroma_db")

_embeddings: HuggingFaceEmbeddings | None = None
_chroma_client: chromadb.PersistentClient | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


def _get_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma_client


def _safe_collection_name(session_id: str) -> str:
    """ChromaDB collection names must be 3-63 chars, alphanumeric + hyphens."""
    name = re.sub(r"[^a-zA-Z0-9-]", "-", session_id)
    return name[:63] if len(name) >= 3 else f"session-{name}"


def ingest_document(
    text: str,
    session_id: str,
    source: str = "reference_doc",
    replace: bool = True,
) -> int:
    """
    Chunk, embed, and store document text under a per-session collection.

    Args:
        text:       Raw extracted text from the uploaded document.
        session_id: Unique ID for the current user session (thread_id from LangGraph).
        source:     Filename or label stored as chunk metadata.
        replace:    If True, clear the session's existing collection first.

    Returns:
        Number of chunks stored.
    """
    collection_name = _safe_collection_name(session_id)
    client = _get_client()

    if replace:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text], metadatas=[{"source": source, "session": session_id}])

    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=_get_embeddings(),
    )
    vectorstore.add_documents(docs)
    return len(docs)


def retrieve_context_with_sources(query: str, session_id: str, k: int = 5) -> tuple[str, list[dict]]:
    """Return (formatted_context, sources) for the RAG chat UI."""
    collection_name = _safe_collection_name(session_id)
    client = _get_client()
    existing = [c.name for c in client.list_collections()]
    if collection_name not in existing:
        return "", []
    vectorstore = Chroma(
        client=client, collection_name=collection_name, embedding_function=_get_embeddings()
    )
    results = vectorstore.similarity_search(query, k=k)
    if not results:
        return "", []
    sources = [
        {"filename": doc.metadata.get("source", "Unknown"), "excerpt": doc.page_content[:150] + "…"}
        for doc in results
    ]
    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', '')}]\n{doc.page_content}" for doc in results
    )
    return context, sources


def retrieve_context(query: str, session_id: str, k: int = 3) -> str:
    """
    Return top-k relevant chunks for the given session, or empty string if none.

    Args:
        query:      The user's current request — used as the similarity search query.
        session_id: Must match the session_id used during ingest_document.
        k:          Number of chunks to retrieve.
    """
    collection_name = _safe_collection_name(session_id)
    client = _get_client()

    existing = [c.name for c in client.list_collections()]
    if collection_name not in existing:
        return ""

    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=_get_embeddings(),
    )

    results = vectorstore.similarity_search(query, k=k)
    if not results:
        return ""

    parts = []
    for doc in results:
        source = doc.metadata.get("source", "")
        parts.append(f"[Source: {source}]\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


def list_session_docs(session_id: str) -> list[str]:
    """Return distinct source filenames indexed for this session."""
    collection_name = _safe_collection_name(session_id)
    client = _get_client()

    existing = [c.name for c in client.list_collections()]
    if collection_name not in existing:
        return []

    collection = client.get_collection(collection_name)
    results = collection.get(include=["metadatas"])
    sources = {m.get("source", "") for m in results["metadatas"] if m}
    return sorted(sources)


def clear_session(session_id: str) -> None:
    """Delete the vectorstore collection for a specific session."""
    collection_name = _safe_collection_name(session_id)
    client = _get_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
