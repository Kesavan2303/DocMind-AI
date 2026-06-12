import streamlit as st
from langchain_core.chat_history import InMemoryChatMessageHistory
from core.chain import generate_chain, analyze_chain, research_chain
from utils.export import to_pdf, to_docx
from utils.parser import parse_pdf, parse_docx
from utils.search import web_search
from utils.db import init_db, save_document, list_documents, get_document, delete_document
from utils.templates import TEMPLATES

init_db()

st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 DocMind AI")
st.caption("Conversational Document Assistant — powered by LangChain + Groq")


# --- Helpers ---

def _is_document(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) >= 400 and not stripped.endswith("?")


def _extract_title(text: str) -> str:
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()[:60]
    return text.strip()[:60]


def _render_export_buttons(text: str) -> None:
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download PDF",
            data=to_pdf(text),
            file_name="document.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            label="Download DOCX",
            data=to_docx(text),
            file_name="document.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )


# --- Session state ---
defaults = {
    "history": InMemoryChatMessageHistory(),
    "messages": [],
    "last_document": None,
    "document_text": None,
    "uploaded_filename": None,
    "pending_message": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# --- Sidebar ---
with st.sidebar:

    # ── Templates ──────────────────────────────────────────
    if not st.session_state.document_text:
        st.header("Document Templates")
        template_names = ["— Select a template —"] + list(TEMPLATES.keys())
        selected = st.selectbox("", template_names, label_visibility="collapsed")

        if selected != "— Select a template —":
            tpl = TEMPLATES[selected]
            st.caption(f"{tpl['icon']}  {tpl['description']}")
            if st.button("Start with this template", use_container_width=True):
                st.session_state.pending_message = tpl["prompt"]
                st.session_state.history = InMemoryChatMessageHistory()
                st.session_state.messages = []
                st.session_state.last_document = None
                st.rerun()

        st.divider()

    # ── Web Research ────────────────────────────────────────
    if not st.session_state.document_text:
        st.header("Web Research")
        web_research_enabled = st.toggle(
            "Enable web research",
            value=False,
            help="Searches the web before generating to include up-to-date information.",
        )
        st.divider()
    else:
        web_research_enabled = False

    # ── Document History ────────────────────────────────────
    st.header("Document History")
    docs = list_documents()

    if not docs:
        st.caption("No saved documents yet.")
    else:
        for doc in docs:
            with st.expander(f"{doc['title'][:35]}  •  {doc['created_at']}"):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Load", key=f"load_{doc['id']}", use_container_width=True):
                        full = get_document(doc["id"])
                        if full:
                            st.session_state.last_document = full["content"]
                            st.session_state.messages = [
                                {"role": "assistant", "content": full["content"]}
                            ]
                            st.session_state.history = InMemoryChatMessageHistory()
                            st.session_state.history.add_ai_message(full["content"])
                            st.rerun()
                with col2:
                    if st.button("Delete", key=f"del_{doc['id']}", use_container_width=True):
                        delete_document(doc["id"])
                        st.rerun()


# --- Chat area ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Attached file indicator ─────────────────────────────────
if st.session_state.document_text:
    col1, col2 = st.columns([0.88, 0.12])
    with col1:
        st.info(f"📎 **{st.session_state.uploaded_filename}** attached — fill, enhance, or ask anything about it")
    with col2:
        if st.button("✕", help="Detach document", use_container_width=True):
            st.session_state.document_text = None
            st.session_state.uploaded_filename = None
            st.session_state.history = InMemoryChatMessageHistory()
            st.session_state.messages = []
            st.session_state.last_document = None
            st.rerun()

# ── Chat input with paperclip attachment ────────────────────
if st.session_state.document_text:
    input_placeholder = "Fill the blanks, expand a section, add a clause, change the tone..."
else:
    input_placeholder = "Tell me what document you need, or attach a file using 📎"

chat = st.chat_input(
    input_placeholder,
    accept_file=True,
    file_type=["pdf", "docx"],
)

# ── Handle submitted input ───────────────────────────────────
user_text = None
if st.session_state.pending_message:
    user_text = st.session_state.pending_message
    st.session_state.pending_message = None
elif chat:
    # Parse an attached file if present
    if chat.files:
        attached = chat.files[0]
        file_bytes = attached.read()
        if attached.name.endswith(".pdf"):
            st.session_state.document_text = parse_pdf(file_bytes)
        else:
            st.session_state.document_text = parse_docx(file_bytes)
        st.session_state.uploaded_filename = attached.name
        st.session_state.history = InMemoryChatMessageHistory()
        st.session_state.messages = []
        st.session_state.last_document = None

    # Use typed text, or a default prompt when only a file was attached
    user_text = chat.text.strip() if chat.text and chat.text.strip() else (
        f"I've attached **{chat.files[0].name}**. What would you like to do with it?"
        if chat.files else None
    )

if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)
    st.session_state.messages.append({"role": "user", "content": user_text})

    with st.chat_message("assistant"):
        if st.session_state.document_text:
            response = st.write_stream(
                analyze_chain.stream({
                    "input": user_text,
                    "history": st.session_state.history.messages,
                    "document": st.session_state.document_text,
                })
            )

        elif web_research_enabled:
            with st.spinner(f'Searching the web for: "{user_text}"...'):
                research_results, sources = web_search(user_text)
            if sources:
                with st.expander(f"Web sources ({len(sources)} found)", expanded=False):
                    for url in sources:
                        st.markdown(f"- {url}")
            response = st.write_stream(
                research_chain.stream({
                    "input": user_text,
                    "history": st.session_state.history.messages,
                    "research": research_results,
                })
            )

        else:
            response = st.write_stream(
                generate_chain.stream({
                    "input": user_text,
                    "history": st.session_state.history.messages,
                })
            )

    st.session_state.history.add_user_message(user_text)
    st.session_state.history.add_ai_message(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    if _is_document(response):
        st.session_state.last_document = response
        title = _extract_title(response)
        save_document(title, response)
        st.toast(f'Saved to history: "{title}"', icon="💾")

if st.session_state.last_document:
    _render_export_buttons(st.session_state.last_document)
