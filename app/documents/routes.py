import uuid
import json
from datetime import datetime
from flask import (
    Blueprint, render_template, request, Response,
    session, jsonify, abort, send_file
)
from flask_login import login_required, current_user
from app.extensions import db
from app.models import Document, Collection
from core.prompts import DOC_TYPE_CONTEXT
from core.graph import stream_graph_response, get_graph_state
from core.chains import get_section_regen_chain, get_enhance_chain
from utils.export import to_pdf, to_docx
from utils.parser import parse_pdf, parse_docx
import io

documents = Blueprint("documents", __name__)

DOC_TYPES = list(DOC_TYPE_CONTEXT.keys())
EDIT_KEYWORDS = {"regenerate", "rewrite", "expand", "improve", "make", "change", "update", "revise", "simplify"}


def _is_section_edit(text: str) -> bool:
    return bool(set(text.lower().split()) & EDIT_KEYWORDS)


def _extract_title(text: str) -> str:
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip()[:80]
    return text.strip()[:80]


# ---------------------------------------------------------------------------
# Generate page
# ---------------------------------------------------------------------------

@documents.route("/generate")
@login_required
def generate():
    if "thread_id" not in session:
        session["thread_id"] = str(uuid.uuid4())
    collections = Collection.query.filter_by(user_id=current_user.id).order_by(Collection.name).all()
    default_col = current_user.default_collection_id
    return render_template(
        "documents/generate.html",
        doc_types=DOC_TYPES,
        collections=collections,
        default_collection_id=default_col,
        thread_id=session["thread_id"],
    )


@documents.route("/new-thread", methods=["POST"])
@login_required
def new_thread():
    session["thread_id"] = str(uuid.uuid4())
    session.pop("current_doc_id", None)
    return jsonify({"thread_id": session["thread_id"]})


# ---------------------------------------------------------------------------
# SSE: document generation stream
# ---------------------------------------------------------------------------

@documents.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    doc_type = data.get("doc_type", DOC_TYPES[0])
    thread_id = data.get("thread_id") or session.get("thread_id") or str(uuid.uuid4())
    collection_id = data.get("collection_id")
    current_doc = data.get("current_document", "")

    session["thread_id"] = thread_id

    def stream():
        full_response = ""

        # Mode A: section edit on existing document
        if current_doc and _is_section_edit(user_message):
            chain = get_section_regen_chain()
            for chunk in chain.stream({
                "doc_type_full": DOC_TYPE_CONTEXT[doc_type]["full_name"],
                "full_document": current_doc,
                "user_request": user_message,
            }):
                full_response += chunk
                yield f"data: {json.dumps({'text': chunk})}\n\n"

            doc_id = _save_document(full_response, doc_type, collection_id)
            yield f"data: {json.dumps({'done': True, 'has_document': True, 'doc_id': doc_id, 'thread_id': thread_id})}\n\n"
            return

        # Mode B: RAG context retrieval
        rag_context = None
        if collection_id:
            try:
                col = Collection.query.filter_by(id=collection_id, user_id=current_user.id).first()
                if col:
                    from core.rag import retrieve_context
                    rag_context = retrieve_context(user_message, session_id=col.chroma_id)
            except Exception:
                pass

        # Mode C: LangGraph multi-agent generation
        for chunk in stream_graph_response(thread_id, user_message, doc_type, rag_context):
            full_response += chunk
            yield f"data: {json.dumps({'text': chunk})}\n\n"

        # Check if document was generated
        state = get_graph_state(thread_id)
        generated = state.get("generated_document")
        if generated:
            doc_id = _save_document(generated, doc_type, collection_id)
            yield f"data: {json.dumps({'done': True, 'has_document': True, 'doc_id': doc_id, 'thread_id': thread_id})}\n\n"
        else:
            yield f"data: {json.dumps({'done': True, 'has_document': False, 'thread_id': thread_id})}\n\n"

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _save_document(content: str, doc_type: str, collection_id=None) -> int:
    title = _extract_title(content)
    doc = Document(
        title=title,
        content=content,
        doc_type=doc_type,
        user_id=current_user.id,
        collection_id=collection_id if collection_id else None,
    )
    db.session.add(doc)
    db.session.commit()
    return doc.id


# ---------------------------------------------------------------------------
# Document enhance stream (for uploaded files)
# ---------------------------------------------------------------------------

@documents.route("/extract-text", methods=["POST"])
@login_required
def extract_text():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file"}), 400
    file_bytes = file.read()
    filename = file.filename or ""
    try:
        text = parse_pdf(file_bytes) if filename.endswith(".pdf") else parse_docx(file_bytes)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@documents.route("/enhance", methods=["POST"])
@login_required
def enhance():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    document_text = data.get("document_text", "")
    history = data.get("history", [])

    from langchain_core.messages import HumanMessage, AIMessage
    messages = []
    for h in history:
        if h["role"] == "user":
            messages.append(HumanMessage(content=h["content"]))
        else:
            messages.append(AIMessage(content=h["content"]))

    def stream():
        full_response = ""
        chain = get_enhance_chain()
        for chunk in chain.stream({"document": document_text, "messages": messages}):
            full_response += chunk
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@documents.route("/history")
@login_required
def history():
    docs = (
        Document.query
        .filter_by(user_id=current_user.id)
        .order_by(Document.created_at.desc())
        .all()
    )
    return render_template("documents/history.html", documents=docs)


@documents.route("/<int:doc_id>")
@login_required
def view(doc_id: int):
    doc = Document.query.filter_by(id=doc_id, user_id=current_user.id).first_or_404()
    return jsonify({"id": doc.id, "title": doc.title, "content": doc.content, "doc_type": doc.doc_type})


@documents.route("/<int:doc_id>/delete", methods=["POST"])
@login_required
def delete(doc_id: int):
    doc = Document.query.filter_by(id=doc_id, user_id=current_user.id).first_or_404()
    db.session.delete(doc)
    db.session.commit()
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@documents.route("/<int:doc_id>/export/pdf")
@login_required
def export_pdf(doc_id: int):
    doc = Document.query.filter_by(id=doc_id, user_id=current_user.id).first_or_404()
    pdf_bytes = to_pdf(doc.content, doc.doc_type)
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{doc.title[:40]}.pdf",
    )


@documents.route("/<int:doc_id>/export/docx")
@login_required
def export_docx(doc_id: int):
    doc = Document.query.filter_by(id=doc_id, user_id=current_user.id).first_or_404()
    docx_bytes = to_docx(doc.content, doc.doc_type)
    return send_file(
        io.BytesIO(docx_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        as_attachment=True,
        download_name=f"{doc.title[:40]}.docx",
    )
