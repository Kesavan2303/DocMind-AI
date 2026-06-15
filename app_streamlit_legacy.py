import uuid
from datetime import datetime

import streamlit as st
from langchain_core.chat_history import InMemoryChatMessageHistory

from utils.langsmith_setup import setup_langsmith
from utils.export import to_pdf, to_docx
from utils.parser import parse_pdf, parse_docx
from utils.search import web_search
from utils.db import init_db, save_document, list_documents, get_document, delete_document
from core.graph import stream_graph_response, get_graph_state, reset_thread
from core.chains import get_enhance_chain, get_section_regen_chain, get_research_chain
from core.prompts import DOC_TYPE_CONTEXT

setup_langsmith()
init_db()

st.set_page_config(page_title="DocMind AI", page_icon="🧠", layout="centered")
st.title("🧠 DocMind AI")
st.caption("Production-ready multi-agent document assistant · LangGraph + Groq")

DOC_TYPES = list(DOC_TYPE_CONTEXT.keys())
SECTION_EDIT_KEYWORDS = {"regenerate", "rewrite", "expand", "improve", "make", "change", "update", "revise", "simplify"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_document(text: str) -> bool:
    s = text.strip()
    return len(s) >= 400 and not s.endswith("?")


def _extract_title(text: str) -> str:
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip()[:60]
    return text.strip()[:60]


def _render_export_buttons(text: str, doc_type: str = "") -> None:
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇ Download PDF", to_pdf(text, doc_type),
            file_name="document.pdf", mime="application/pdf",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "⬇ Download DOCX", to_docx(text, doc_type),
            file_name="document.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )


def _is_section_edit(text: str) -> bool:
    words = set(text.lower().split())
    return bool(words & SECTION_EDIT_KEYWORDS)


def _new_thread() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_defaults = {
    "thread_id": _new_thread(),
    "messages": [],
    "doc_type": DOC_TYPES[0],
    "generated_document": None,
    "document_text": None,
    "uploaded_filename": None,
    "rag_ingested": False,
    "rag_filename": None,
    "enhance_history": None,
    "sessions": [],
    "web_research_on": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:

    # ── Document type ──────────────────────────────────────────
    st.header("Document Type")
    new_doc_type = st.selectbox(
        "", DOC_TYPES, index=DOC_TYPES.index(st.session_state.doc_type),
        label_visibility="collapsed",
        help="Selects the document template and tailors the agent's questions.",
    )
    if new_doc_type != st.session_state.doc_type:
        st.session_state.doc_type = new_doc_type
        st.session_state.messages = []
        st.session_state.generated_document = None
        st.session_state.thread_id = _new_thread()

    if st.button("🆕 New Document", use_container_width=True):
        st.session_state.messages = []
        st.session_state.generated_document = None
        st.session_state.thread_id = _new_thread()
        st.rerun()

    st.divider()

    # ── Reference doc for RAG ──────────────────────────────────
    st.header("Reference Document (RAG)")
    st.caption("Upload a reference PDF/DOCX. The writer will ground the output in its content.")
    ref_file = st.file_uploader(
        "Upload reference doc",
        type=["pdf", "docx"],
        key="rag_uploader",
        label_visibility="collapsed",
    )
    if ref_file and ref_file.name != st.session_state.rag_filename:
        with st.spinner("Indexing reference document…"):
            from core.rag import ingest_document
            file_bytes = ref_file.read()
            raw_text = (
                parse_pdf(file_bytes) if ref_file.name.endswith(".pdf")
                else parse_docx(file_bytes)
            )
            n_chunks = ingest_document(
                raw_text,
                session_id=st.session_state.thread_id,
                source=ref_file.name,
            )
        st.session_state.rag_ingested = True
        st.session_state.rag_filename = ref_file.name
        st.success(f"Indexed {n_chunks} chunks from {ref_file.name}")

    if st.session_state.rag_ingested:
        st.caption(f"📚 RAG active: **{st.session_state.rag_filename}**")
        if st.button("Clear reference doc", use_container_width=True):
            from core.rag import clear_session
            clear_session(st.session_state.thread_id)
            st.session_state.rag_ingested = False
            st.session_state.rag_filename = None
            st.rerun()

    st.divider()

    # ── Web Research ───────────────────────────────────────────
    if not st.session_state.document_text:
        st.header("Web Research")
        st.session_state.web_research_on = st.toggle(
            "Enable before generation",
            value=st.session_state.web_research_on,
            help="Searches the web and injects results before generating.",
        )
        st.divider()

    # ── Session history ────────────────────────────────────────
    st.header("Session History")
    db_docs = list_documents()
    if not db_docs:
        st.caption("No saved documents yet.")
    else:
        for doc in db_docs:
            with st.expander(f"{doc['title'][:32]}  •  {doc['created_at']}"):
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Load", key=f"load_{doc['id']}", use_container_width=True):
                        full = get_document(doc["id"])
                        if full:
                            st.session_state.generated_document = full["content"]
                            st.session_state.messages = [
                                {"role": "assistant", "content": full["content"]}
                            ]
                            st.rerun()
                with c2:
                    if st.button("Delete", key=f"del_{doc['id']}", use_container_width=True):
                        delete_document(doc["id"])
                        st.rerun()

    st.divider()
    st.markdown("### Previous sessions")
    for s in reversed(st.session_state.sessions[-10:]):
        st.markdown(f"**{s['doc_type']}** · {s['timestamp']}")
        st.caption(s["preview"])


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Attached document indicator ────────────────────────────
if st.session_state.document_text:
    col1, col2 = st.columns([0.88, 0.12])
    with col1:
        st.info(f"📎 **{st.session_state.uploaded_filename}** — fill, enhance, or ask anything about it")
    with col2:
        if st.button("✕", help="Detach document", use_container_width=True):
            st.session_state.document_text = None
            st.session_state.uploaded_filename = None
            st.session_state.enhance_history = None
            st.session_state.messages = []
            st.session_state.generated_document = None
            st.rerun()

# ── Chat input with paperclip attachment ───────────────────
placeholder = (
    "Fill the blanks, expand a section, add a clause, change the tone…"
    if st.session_state.document_text
    else f"Describe what you need ({st.session_state.doc_type}) or attach a file via 📎"
)

chat = st.chat_input(placeholder, accept_file=True, file_type=["pdf", "docx"])

# ── Resolve user input ─────────────────────────────────────
user_text: str | None = None

if chat:
    if chat.files:
        attached = chat.files[0]
        file_bytes = attached.read()
        st.session_state.document_text = (
            parse_pdf(file_bytes) if attached.name.endswith(".pdf")
            else parse_docx(file_bytes)
        )
        st.session_state.uploaded_filename = attached.name
        st.session_state.enhance_history = InMemoryChatMessageHistory()
        st.session_state.messages = []
        st.session_state.generated_document = None

    user_text = (
        chat.text.strip() if chat.text and chat.text.strip()
        else (f"I've attached **{chat.files[0].name}**. What would you like to do with it?"
              if chat.files else None)
    )

if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)
    st.session_state.messages.append({"role": "user", "content": user_text})

    # ── Routing ───────────────────────────────────────────────
    with st.chat_message("assistant"):

        # Mode A: Enhance an uploaded document
        if st.session_state.document_text:
            if st.session_state.enhance_history is None:
                st.session_state.enhance_history = InMemoryChatMessageHistory()

            st.session_state.enhance_history.add_user_message(user_text)
            chain = get_enhance_chain()
            response = st.write_stream(
                chain.stream({
                    "document": st.session_state.document_text,
                    "messages": st.session_state.enhance_history.messages,
                })
            )
            st.session_state.enhance_history.add_ai_message(response)

        # Mode B: Section-level edit on an already-generated document
        elif st.session_state.generated_document and _is_section_edit(user_text):
            chain = get_section_regen_chain()
            response = st.write_stream(
                chain.stream({
                    "doc_type_full": DOC_TYPE_CONTEXT[st.session_state.doc_type]["full_name"],
                    "full_document": st.session_state.generated_document,
                    "user_request": user_text,
                })
            )
            st.session_state.generated_document = response

        # Mode C: Web-research-grounded generation (plain LCEL, no graph)
        elif st.session_state.web_research_on and not st.session_state.generated_document:
            with st.spinner(f'Searching the web for: "{user_text}"…'):
                research_results, sources = web_search(user_text)
            if sources:
                with st.expander(f"Web sources ({len(sources)} found)", expanded=False):
                    for url in sources:
                        st.markdown(f"- {url}")
            chain = get_research_chain()
            response = st.write_stream(
                chain.stream({
                    "input": user_text,
                    "history": [],
                    "research": research_results,
                })
            )

        # Mode D: LangGraph multi-agent document generation
        else:
            rag_context: str | None = None
            if st.session_state.rag_ingested:
                from core.rag import retrieve_context
                with st.spinner("Retrieving relevant context from reference doc…"):
                    rag_context = retrieve_context(user_text, session_id=st.session_state.thread_id)

            response = st.write_stream(
                stream_graph_response(
                    thread_id=st.session_state.thread_id,
                    user_message=user_text,
                    doc_type=st.session_state.doc_type,
                    retrieved_context=rag_context,
                )
            )

            # Sync generated_document from graph state
            state = get_graph_state(st.session_state.thread_id)
            if state.get("generated_document"):
                st.session_state.generated_document = state["generated_document"]

    st.session_state.messages.append({"role": "assistant", "content": response})

    # ── Auto-save to DB and session list ──────────────────────
    if _is_document(response) and response != st.session_state.get("_last_saved"):
        title = _extract_title(response)
        save_document(title, response)
        st.session_state._last_saved = response
        st.session_state.sessions.append({
            "doc_type": st.session_state.doc_type,
            "timestamp": datetime.now().strftime("%d %b %H:%M"),
            "preview": response.strip()[:80] + "…",
        })
        st.toast(f'Saved: "{title}"', icon="💾")

# ── Export buttons ─────────────────────────────────────────
if st.session_state.generated_document:
    _render_export_buttons(
        st.session_state.generated_document,
        doc_type=DOC_TYPE_CONTEXT[st.session_state.doc_type]["full_name"],
    )
elif (
    st.session_state.document_text
    and st.session_state.messages
    and st.session_state.messages[-1]["role"] == "assistant"
    and _is_document(st.session_state.messages[-1]["content"])
):
    _render_export_buttons(st.session_state.messages[-1]["content"])
