from typing import Annotated, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AIMessageChunk
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from core.prompts import DOC_TYPE_CONTEXT


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class DocMindState(TypedDict):
    messages: Annotated[list, add_messages]
    doc_type: str
    requirements_complete: bool
    retrieved_context: Optional[str]
    generated_document: Optional[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def requirements_node(state: DocMindState) -> dict:
    from core.chains import get_requirements_chain

    doc_type = state.get("doc_type", "PRD")
    ctx = DOC_TYPE_CONTEXT.get(doc_type, DOC_TYPE_CONTEXT["PRD"])

    chain = get_requirements_chain()
    response = "".join(
        chain.stream({
            "doc_type_full": ctx["full_name"],
            "key_questions": ctx["key_questions"],
            "key_sections": ctx["key_sections"],
            "messages": state["messages"],
        })
    )

    if "REQUIREMENTS_COMPLETE" in response:
        return {"requirements_complete": True}

    return {
        "messages": [AIMessage(content=response)],
        "requirements_complete": False,
    }


def writer_node(state: DocMindState) -> dict:
    from core.chains import get_writer_chain

    doc_type = state.get("doc_type", "PRD")
    ctx = DOC_TYPE_CONTEXT.get(doc_type, DOC_TYPE_CONTEXT["PRD"])

    req_summary = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
        for m in state["messages"]
    )

    rag_ctx = state.get("retrieved_context") or ""
    rag_section = (
        f"\n\nReference document context (use to ground the document):\n{rag_ctx}"
        if rag_ctx else ""
    )

    chain = get_writer_chain()
    response = "".join(
        chain.stream({
            "doc_type_full": ctx["full_name"],
            "key_sections": ctx["key_sections"],
            "requirements_summary": req_summary,
            "rag_context_section": rag_section,
            "research_context_section": "",
        })
    )

    return {
        "messages": [AIMessage(content=response)],
        "generated_document": response,
    }


# ---------------------------------------------------------------------------
# Supervisor router
# ---------------------------------------------------------------------------

def _supervisor(state: DocMindState) -> str:
    if state.get("requirements_complete", False):
        return "writer_agent"
    return "requirements_agent"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

_builder = StateGraph(DocMindState)
_builder.add_node("requirements_agent", requirements_node)
_builder.add_node("writer_agent", writer_node)
_builder.set_conditional_entry_point(_supervisor)

# After requirements_agent: if now complete → writer, else → END (wait for next user turn)
_builder.add_conditional_edges(
    "requirements_agent",
    _supervisor,
    {"writer_agent": "writer_agent", "requirements_agent": END},
)
_builder.add_edge("writer_agent", END)

_checkpointer = MemorySaver()
docmind_graph = _builder.compile(checkpointer=_checkpointer)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def stream_graph_response(
    thread_id: str,
    user_message: str,
    doc_type: str,
    retrieved_context: Optional[str] = None,
):
    """Yield text chunks from the graph for the given user turn."""
    config = {"configurable": {"thread_id": thread_id}}
    input_state: dict = {
        "messages": [HumanMessage(content=user_message)],
        "doc_type": doc_type,
    }
    if retrieved_context is not None:
        input_state["retrieved_context"] = retrieved_context

    for chunk, metadata in docmind_graph.stream(
        input_state,
        config=config,
        stream_mode="messages",
    ):
        if (
            isinstance(chunk, AIMessageChunk)
            and chunk.content
            and metadata.get("langgraph_node") in ("requirements_agent", "writer_agent")
        ):
            yield chunk.content


def get_graph_state(thread_id: str) -> dict:
    """Return the current persisted state values for a thread."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = docmind_graph.get_state(config)
    return snapshot.values if snapshot else {}


def reset_thread(thread_id: str, doc_type: str) -> None:
    """Seed a fresh state for a new document session on an existing thread_id."""
    config = {"configurable": {"thread_id": thread_id}}
    docmind_graph.update_state(
        config,
        {
            "messages": [],
            "doc_type": doc_type,
            "requirements_complete": False,
            "retrieved_context": None,
            "generated_document": None,
        },
    )
