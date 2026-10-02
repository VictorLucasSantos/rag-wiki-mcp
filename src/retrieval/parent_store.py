"""
src/retrieval/parent_store.py — Docstore dos chunks "pai" (estratégia parent_child).

Os filhos (pequenos) vão para o vector store; os pais (grandes) ficam aqui.
No retrieval, cada filho encontrado é trocado pelo seu pai, e pais repetidos
são deduplicados — o LLM recebe contexto rico sem redundância.

Um JSON simples basta para bases pequenas/médias; em produção use Redis,
Postgres ou o próprio payload do vector store.
"""

import json
from pathlib import Path

from src.ingestion.chunker import Chunk


class ParentStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self._parents: dict[str, dict] | None = None

    @property
    def parents(self) -> dict[str, dict]:
        if self._parents is None:
            self._parents = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        return self._parents

    def add(self, parents: list[Chunk]) -> None:
        for p in parents:
            self.parents[p.chunk_id] = {"text": p.text, "metadata": p.metadata}

    def delete_document(self, doc_id: str) -> None:
        for pid in [pid for pid, p in self.parents.items() if p["metadata"].get("doc_id") == doc_id]:
            del self.parents[pid]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.parents, ensure_ascii=False), encoding="utf-8")

    def expand(self, docs: list[dict]) -> list[dict]:
        """Troca filhos pelos pais, preservando a ordem de relevância."""
        expanded, seen = [], set()
        for doc in docs:
            parent_id = doc.get("metadata", {}).get("parent_id")
            if not parent_id or parent_id not in self.parents:
                expanded.append(doc)
                continue
            if parent_id in seen:
                continue
            seen.add(parent_id)
            parent = self.parents[parent_id]
            expanded.append({**doc, "id": parent_id, "text": parent["text"], "matched_child": doc["text"]})
        return expanded
