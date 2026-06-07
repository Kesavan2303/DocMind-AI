import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 DocMind AI")
st.caption("Conversational Document Generation Agent — powered by LangChain + Groq")

# ── LLM & Chain setup ────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are DocMind AI, an expert document generator.
    When the user tells you what document they need:
    - Ask clarifying questions ONE BY ONE (max 3 questions)
    - Once you have enough context, generate the full professional document
    - After generating, ask if they want to refine any section"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

# ── Session state — memory persists across reruns ────────
if "history" not in st.session_state:
    st.session_state.history = InMemoryChatMessageHistory()

if "messages" not in st.session_state:
    st.session_state.messages = []  # for rendering chat bubbles

# ── Render existing chat history ─────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ───────────────────────────────────────────
if user_input := st.chat_input("Tell me what document you need..."):

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Stream AI response
    with st.chat_message("assistant"):
        response = st.write_stream(
            chain.stream({
                "input": user_input,
                "history": st.session_state.history.messages
            })
        )

    # Save to memory
    st.session_state.history.add_user_message(user_input)
    st.session_state.history.add_ai_message(response)
    st.session_state.messages.append({"role": "assistant", "content": response})