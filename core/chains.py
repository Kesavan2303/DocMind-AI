import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from pydantic import SecretStr
from core.prompts import (
    requirements_prompt,
    writer_prompt,
    section_regen_prompt,
    enhance_prompt,
    research_prompt,
    rag_summary_prompt,
)

load_dotenv()


def _llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=SecretStr(os.getenv("GROQ_API_KEY") or ""),
        streaming=True,
    )


def get_requirements_chain() -> Runnable:
    return requirements_prompt | _llm() | StrOutputParser()


def get_writer_chain() -> Runnable:
    return writer_prompt | _llm() | StrOutputParser()


def get_section_regen_chain() -> Runnable:
    return section_regen_prompt | _llm() | StrOutputParser()


def get_enhance_chain() -> Runnable:
    return enhance_prompt | _llm() | StrOutputParser()


def get_research_chain() -> Runnable:
    return research_prompt | _llm() | StrOutputParser()


def get_rag_summary_chain() -> Runnable:
    return rag_summary_prompt | _llm() | StrOutputParser()
