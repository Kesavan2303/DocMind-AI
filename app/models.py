from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app.extensions import db


class User(db.Model, UserMixin):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    default_collection_id = db.Column(db.Integer, db.ForeignKey("collections.id"), nullable=True)

    collections = db.relationship(
        "Collection", foreign_keys="Collection.user_id",
        backref="owner", lazy="dynamic", cascade="all, delete-orphan"
    )
    documents = db.relationship("Document", backref="author", lazy="dynamic", cascade="all, delete-orphan")

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.email}>"


class Collection(db.Model):
    __tablename__ = "collections"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    description = db.Column(db.String(500), default="")
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    files = db.relationship(
        "CollectionFile", backref="collection",
        lazy="dynamic", cascade="all, delete-orphan"
    )
    documents = db.relationship("Document", backref="collection", lazy="dynamic")

    @property
    def chroma_id(self) -> str:
        """Deterministic ChromaDB collection name scoped to this user + collection."""
        return f"u{self.user_id}c{self.id}"

    @property
    def file_count(self) -> int:
        return self.files.count()

    @property
    def chunk_count(self) -> int:
        return sum(f.chunk_count for f in self.files)

    def __repr__(self):
        return f"<Collection {self.name}>"


class CollectionFile(db.Model):
    __tablename__ = "collection_files"

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    chunk_count = db.Column(db.Integer, default=0)
    collection_id = db.Column(db.Integer, db.ForeignKey("collections.id"), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)


class Document(db.Model):
    __tablename__ = "documents"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(256), nullable=False)
    content = db.Column(db.Text, nullable=False)
    doc_type = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    collection_id = db.Column(db.Integer, db.ForeignKey("collections.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Document {self.title}>"
