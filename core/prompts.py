from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """You are DocMind AI, an expert document generator.
    When the user tells you what document they need:
    - Ask clarifying questions ONE BY ONE (max 3 questions)
    - Once you have enough context, generate the full professional document
    - After generating, ask if they want to refine any section"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
