import streamlit as st
from langchain_core.chat_history import InMemoryChatMessageHistory
from core.chain import chain
from utils.export import to_pdf, to_docx

st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 DocMind AI")
st.caption("Conversational Document Generation Agent — powered by LangChain + Groq")


def _is_document(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) >= 400 and not stripped.endswith("?")


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


if "history" not in st.session_state:
    st.session_state.history = InMemoryChatMessageHistory()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_document" not in st.session_state:
    st.session_state.last_document = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Tell me what document you need..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        response = st.write_stream(
            chain.stream({
                "input": user_input,
                "history": st.session_state.history.messages,
            })
        )

    st.session_state.history.add_user_message(user_input)
    st.session_state.history.add_ai_message(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    if _is_document(response):
        st.session_state.last_document = response

if st.session_state.last_document:
    _render_export_buttons(st.session_state.last_document)
