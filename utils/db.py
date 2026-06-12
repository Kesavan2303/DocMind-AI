import sqlite3
import os
from datetime import datetime

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "docmind.db")


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    return sqlite3.connect(_DB_PATH)


def init_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def save_document(title: str, content: str) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO documents (title, content, created_at) VALUES (?, ?, ?)",
            (title, content, datetime.now().strftime("%Y-%m-%d %H:%M")),
        )
        return cur.lastrowid


def list_documents() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, title, created_at FROM documents ORDER BY created_at DESC"
        ).fetchall()
    return [{"id": r[0], "title": r[1], "created_at": r[2]} for r in rows]


def get_document(doc_id: int) -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT id, title, content, created_at FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()
    if row:
        return {"id": row[0], "title": row[1], "content": row[2], "created_at": row[3]}
    return None


def delete_document(doc_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
