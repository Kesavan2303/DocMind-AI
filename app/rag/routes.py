import json
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, Response
from flask_login import login_required, current_user
from app.extensions import db
from app.models import Collection, CollectionFile, User
from utils.parser import parse_pdf, parse_docx

rag = Blueprint("rag", __name__)


# ---------------------------------------------------------------------------
# Collections list + create
# ---------------------------------------------------------------------------

@rag.route("/collections")
@login_required
def collections():
    cols = Collection.query.filter_by(user_id=current_user.id).order_by(Collection.created_at.desc()).all()
    return render_template("rag/collections.html", collections=cols,
                           default_id=current_user.default_collection_id)


@rag.route("/collections/create", methods=["POST"])
@login_required
def create_collection():
    name = request.form.get("name", "").strip()
    description = request.form.get("description", "").strip()
    if not name:
        flash("Collection name is required.", "error")
        return redirect(url_for("rag.collections"))
    col = Collection(name=name, description=description, user_id=current_user.id)
    db.session.add(col)
    db.session.commit()
    flash(f'Collection "{name}" created.', "success")
    return redirect(url_for("rag.collection_detail", col_id=col.id))


@rag.route("/collections/<int:col_id>")
@login_required
def collection_detail(col_id: int):
    col = Collection.query.filter_by(id=col_id, user_id=current_user.id).first_or_404()
    files = col.files.order_by(CollectionFile.uploaded_at.desc()).all()
    return render_template("rag/collection_detail.html", collection=col, files=files,
                           is_default=(current_user.default_collection_id == col.id))


@rag.route("/collections/<int:col_id>/delete", methods=["POST"])
@login_required
def delete_collection(col_id: int):
    col = Collection.query.filter_by(id=col_id, user_id=current_user.id).first_or_404()
    from core.rag import clear_session
    clear_session(col.chroma_id)
    db.session.delete(col)
    db.session.commit()
    flash(f'Collection "{col.name}" deleted.', "success")
    return redirect(url_for("rag.collections"))


@rag.route("/collections/<int:col_id>/set-default", methods=["POST"])
@login_required
def set_default(col_id: int):
    col = Collection.query.filter_by(id=col_id, user_id=current_user.id).first_or_404()
    current_user.default_collection_id = col.id
    db.session.commit()
    return jsonify({"success": True, "message": f'"{col.name}" set as default collection.'})


# ---------------------------------------------------------------------------
# File upload into collection
# ---------------------------------------------------------------------------

@rag.route("/collections/<int:col_id>/upload", methods=["POST"])
@login_required
def upload_file(col_id: int):
    col = Collection.query.filter_by(id=col_id, user_id=current_user.id).first_or_404()
    file = request.files.get("file")

    if not file or not file.filename:
        flash("No file selected.", "error")
        return redirect(url_for("rag.collection_detail", col_id=col_id))

    filename = file.filename
    file_bytes = file.read()

    try:
        raw_text = (
            parse_pdf(file_bytes) if filename.endswith(".pdf")
            else parse_docx(file_bytes)
        )
    except Exception as e:
        flash(f"Could not parse file: {e}", "error")
        return redirect(url_for("rag.collection_detail", col_id=col_id))

    from core.rag import ingest_document
    n_chunks = ingest_document(raw_text, session_id=col.chroma_id, source=filename, replace=False)

    col_file = CollectionFile(filename=filename, chunk_count=n_chunks, collection_id=col.id)
    db.session.add(col_file)
    db.session.commit()

    flash(f'"{filename}" indexed — {n_chunks} chunks added to "{col.name}".', "success")
    return redirect(url_for("rag.collection_detail", col_id=col_id))


@rag.route("/collections/<int:col_id>/files/<int:file_id>/delete", methods=["POST"])
@login_required
def delete_file(col_id: int, file_id: int):
    col = Collection.query.filter_by(id=col_id, user_id=current_user.id).first_or_404()
    f = CollectionFile.query.filter_by(id=file_id, collection_id=col.id).first_or_404()
    db.session.delete(f)
    db.session.commit()
    flash(f'"{f.filename}" removed from collection.', "success")
    return redirect(url_for("rag.collection_detail", col_id=col_id))


# ---------------------------------------------------------------------------
# RAG Chat
# ---------------------------------------------------------------------------

@rag.route("/chat")
@login_required
def chat_page():
    cols = Collection.query.filter_by(user_id=current_user.id).order_by(Collection.name).all()
    default_id = current_user.default_collection_id
    return render_template("rag/chat.html", collections=cols, default_id=default_id)


@rag.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json()
    query = data.get("query", "").strip()
    collection_id = data.get("collection_id")

    if not query or not collection_id:
        return jsonify({"error": "Query and collection are required."}), 400

    col = Collection.query.filter_by(id=collection_id, user_id=current_user.id).first_or_404()

    def stream():
        from core.rag import retrieve_context_with_sources
        from core.chains import get_rag_summary_chain

        context, sources = retrieve_context_with_sources(query, session_id=col.chroma_id)

        if not context:
            msg = "No relevant information found in this collection for your query."
            yield f"data: {json.dumps({'text': msg})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"
            return

        chain = get_rag_summary_chain()
        for chunk in chain.stream({"query": query, "context": context}):
            yield f"data: {json.dumps({'text': chunk})}\n\n"

        yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
