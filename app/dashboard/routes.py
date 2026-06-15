from flask import Blueprint, render_template
from flask_login import login_required, current_user
from app.models import Document, Collection

dashboard = Blueprint("dashboard", __name__)


@dashboard.route("/")
@login_required
def index():
    doc_count = Document.query.filter_by(user_id=current_user.id).count()
    collection_count = Collection.query.filter_by(user_id=current_user.id).count()
    recent_docs = (
        Document.query
        .filter_by(user_id=current_user.id)
        .order_by(Document.created_at.desc())
        .limit(5)
        .all()
    )
    return render_template(
        "dashboard/index.html",
        doc_count=doc_count,
        collection_count=collection_count,
        recent_docs=recent_docs,
    )
