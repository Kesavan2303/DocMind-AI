import os


def setup_langsmith() -> None:
    api_key = os.getenv("LANGSMITH_API_KEY", "")
    if not api_key:
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "DocMind-AI")
    os.environ["LANGCHAIN_API_KEY"] = api_key
