import streamlit as st
from langchain_core.chat_history import InMemoryChatMessageHistory
from core.chain import chain

st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 DocMind AI")
st.caption("Conversational Document Generation Agent — powered by LangChain + Groq")

if "history" not in st.session_state:
    st.session_state.history = InMemoryChatMessageHistory()

if "messages" not in st.session_state:
    st.session_state.messages = []

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
                "history": st.session_state.history.messages
            })
        )

    st.session_state.history.add_user_message(user_input)
    st.session_state.history.add_ai_message(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
