import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)   

# Modern memory — replaces ConversationBufferWindowMemory
history = InMemoryChatMessageHistory()

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are DocMind AI, an expert document generator.
    When user tells you what document they need:
    - Ask clarifying questions ONE BY ONE (max 3 questions)
    - Once you have enough context, generate the full professional document
    - After generating, ask if they want to refine any section"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

print("🧠 DocMind AI — Tell me what document you need!\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    response = chain.invoke({
        "input": user_input,
        "history": history.messages
    })

    # Save to memory
    history.add_user_message(user_input)
    history.add_ai_message(response)

    print(f"\nDocMind: {response}\n")