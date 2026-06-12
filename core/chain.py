import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr
from core.prompts import generate_prompt, analyze_prompt, research_prompt

load_dotenv()

os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "docmind-ai"))

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=SecretStr(os.getenv("GROQ_API_KEY") or "")
)

generate_chain = generate_prompt | llm | StrOutputParser()
analyze_chain = analyze_prompt | llm | StrOutputParser()
research_chain = research_prompt | llm | StrOutputParser()
