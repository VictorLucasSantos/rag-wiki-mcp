"""
src/ingestion/cleaner.py — Fase 2: Limpeza e pré-processamento.

Responsabilidade:
  Remover ruído que atrapalha embeddings e BM25 sem perder conteúdo:
    - cabeçalhos/rodapés repetidos em todas as páginas (PDFs)
    - números de página soltos
    - hifenização de fim de linha ("informa-\\nção" → "informação")
    - caracteres de controle e espaços duplicados
    - excesso de linhas em branco

Por que isso importa:
  Um rodapé "Empresa X — Confidencial — Página 3" repetido em todo chunk
  faz todos os chunks parecerem similares entre si e polui o BM25.
"""

import re
import unicodedata
from collections import Counter

from src.ingestion.loader import Document, content_hash

_PAGE_NUMBER = re.compile(r"^\s*(p[áa]gina\s*)?\d{1,4}(\s*(de|/)\s*\d{1,4})?\s*$", re.IGNORECASE)
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _repeated_lines(text: str, min_pages: int = 3, ratio: float = 0.5, edge_lines: int = 2) -> set[str]:
    """
    Detecta linhas que se repetem em boa parte das páginas (cabeçalho/rodapé).
    Só faz sentido para documentos paginados (PDF com marcadores [Página N]).
    """
    pages = re.split(r"^\[Página \d+\]$", text, flags=re.MULTILINE)
    pages = [p for p in pages if p.strip()]
    if len(pages) < min_pages:
        return set()

    counts = Counter()
    for page in pages:
        # Só as bordas da página podem ser cabeçalho/rodapé — evita apagar
        # frases do corpo que por acaso se repetem entre páginas.
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        counts.update(set(lines[:edge_lines] + lines[-edge_lines:]))

    threshold = max(min_pages, int(len(pages) * ratio))
    return {line for line, n in counts.items() if n >= threshold and len(line) < 120}


def clean_text(text: str) -> str:
    """Aplica todas as regras de limpeza em um texto."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = text.replace("\u00a0", " ")  # espaço não separável

    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    repeated = _repeated_lines(text)
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped in repeated or _PAGE_NUMBER.match(stripped):
            continue
        lines.append(line.rstrip())
    text = "\n".join(lines)

    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_document(doc: Document) -> Document:
    """Retorna o documento limpo. O content_hash continua sendo o do arquivo original."""
    cleaned = clean_text(doc.content)
    # Mantém os marcadores de página para o LLM poder citar a página,
    # mas descarta marcadores que ficaram sem conteúdo.
    cleaned = re.sub(r"(\[Página \d+\]\n*)+(?=\[Página \d+\])", "", cleaned)
    return Document(content=cleaned, metadata={**doc.metadata, "clean_hash": content_hash(cleaned)})
