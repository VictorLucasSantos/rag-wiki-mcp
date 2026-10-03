"""
app/conversations.py — Histórico de conversas da interface web (SQLite).

Cada mensagem do assistente guarda junto, em `meta`, o que a tela precisa para
reabrir a conversa como ela foi: fontes (com o trecho), modelo, opções de busca,
variações do multi-query e o documento do HyDE.

SQLite é da biblioteca padrão e aguenta tranquilamente o uso de uma pessoa.
Uma conexão por operação: o servidor grava a partir de threads diferentes.
"""

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,          -- "user" | "assistant"
    content         TEXT NOT NULL,
    meta            TEXT,                   -- JSON
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_title(text: str, limit: int = 60) -> str:
    title = " ".join(text.split())
    return title if len(title) <= limit else title[: limit - 1].rstrip() + "…"


def history_from(conv: dict, turns: int) -> list[dict]:
    """
    As últimas `turns` trocas completas (pergunta + resposta) de uma conversa,
    no formato que RAGPipeline.ask espera. Respostas que foram erro ficam de fora
    junto com a pergunta delas.
    """
    pairs, pending = [], None
    for m in conv.get("messages", []):
        if m["role"] == "user":
            pending = m["content"]
        elif pending is not None and not m.get("meta", {}).get("error"):
            pairs.append([{"role": "user", "content": pending}, {"role": "assistant", "content": m["content"]}])
            pending = None
    return [msg for pair in (pairs[-turns:] if turns > 0 else []) for msg in pair]


class ConversationStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def create(self, title: str) -> dict:
        conv = {"id": uuid.uuid4().hex, "title": make_title(title) or "Nova conversa",
                "created_at": _now(), "updated_at": _now()}
        with closing(self._connect()) as db, db:
            db.execute("INSERT INTO conversations VALUES (:id, :title, :created_at, :updated_at)", conv)
        return conv

    def list(self) -> list[dict]:
        with closing(self._connect()) as db:
            rows = db.execute("""
                SELECT c.*, COUNT(m.id) AS messages
                FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
                GROUP BY c.id ORDER BY c.updated_at DESC
            """).fetchall()
        return [dict(r) for r in rows]

    def get(self, conv_id: str) -> dict | None:
        with closing(self._connect()) as db:
            conv = db.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
            if not conv:
                return None
            rows = db.execute(
                "SELECT role, content, meta, created_at FROM messages WHERE conversation_id = ? ORDER BY id",
                (conv_id,),
            ).fetchall()
        messages = [{**dict(r), "meta": json.loads(r["meta"]) if r["meta"] else {}} for r in rows]
        return {**dict(conv), "messages": messages}

    def add_message(self, conv_id: str, role: str, content: str, meta: dict | None = None) -> None:
        now = _now()
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO messages (conversation_id, role, content, meta, created_at) VALUES (?, ?, ?, ?, ?)",
                (conv_id, role, content, json.dumps(meta, ensure_ascii=False) if meta else None, now),
            )
            db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))

    def rename(self, conv_id: str, title: str) -> bool:
        with closing(self._connect()) as db, db:
            cur = db.execute("UPDATE conversations SET title = ? WHERE id = ?", (make_title(title, 120), conv_id))
        return cur.rowcount > 0

    def delete(self, conv_id: str) -> bool:
        with closing(self._connect()) as db, db:
            cur = db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        return cur.rowcount > 0
