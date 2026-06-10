import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr
from core.prompts import prompt

load_dotenv()

# LangSmith tracing — enabled via LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY env vars
os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "docmind-ai"))

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=SecretStr(os.getenv("GROQ_API_KEY") or "")
)

chain = prompt | llm | StrOutputParser()
