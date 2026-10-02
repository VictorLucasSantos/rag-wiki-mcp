"""
src/ingestion/chunker.py — Fase 3: Chunking (divisão em pedaços).

Esta é a camada que MAIS afeta a qualidade do RAG e que MAIS é negligenciada.

O retriever só pode recuperar o que foi indexado.
Se os chunks não capturarem as unidades de significado corretas,
o melhor embedding do mundo não vai recuperar o contexto relevante.

TRADE-OFF fundamental:
  Chunks pequenos → alta precisão de busca, menos contexto por chunk
  Chunks grandes  → mais contexto, menor precisão de busca

Estratégias disponíveis:
  1. RecursiveChunker      → baseline, simples, funciona bem
  2. MarkdownChunker       → respeita estrutura de headers (document-aware)
  3. ParentChildChunker    → busca em chunks pequenos, entrega chunks grandes
  4. AutoChunker           → escolhe 1 ou 2 conforme o tipo de arquivo

Todas as estratégias retornam (chunks, parents):
  chunks  → vão para o vector store e para o BM25
  parents → só o ParentChildChunker preenche; ficam num docstore à parte
"""

import hashlib
from dataclasses import dataclass, field

from src.ingestion.loader import Document

SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def _splitter(chunk_size: int, chunk_overlap: int):
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        raise ImportError("pip install langchain-text-splitters")
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )


def doc_key(document: Document) -> str:
    """
    Identificador estável do documento, derivado do caminho de origem.

    IDs determinísticos permitem upsert idempotente e remoção dos chunks
    antigos quando o documento muda (indexação incremental).
    """
    source = document.metadata.get("source", document.metadata.get("file_name", "doc"))
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]


@dataclass
class Chunk:
    """
    Representa um pedaço de texto pronto para ser indexado.

    Atributos:
        text     : conteúdo do chunk (é o que o LLM lê)
        metadata : herda metadados do documento + posição/seção
        chunk_id : identificador único e determinístico
        context  : snippet do Contextual Retrieval (opcional) — situa o chunk
                   no documento. Entra no embedding e no BM25, não no prompt.
    """
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""
    context: str = ""

    @property
    def indexed_text(self) -> str:
        """Texto usado para embedding e BM25."""
        return f"{self.context}\n\n{self.text}" if self.context else self.text

    def __repr__(self):
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(id={self.chunk_id}, preview='{preview}...')"


class RecursiveChunker:
    """
    Estratégia 1: Fixed-size chunking com separadores naturais.

    Como funciona:
      Tenta dividir em separadores "naturais" (parágrafos → frases → espaços)
      antes de cortar no tamanho fixo. O overlap reduz o risco de cortar
      uma ideia ao meio na fronteira — é um paliativo, não uma solução.

    Quando usar:
      - Baseline para qualquer tipo de documento
      - Documentação geral, emails, textos corridos
    """

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200, **_):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> tuple[list[Chunk], list[Chunk]]:
        key = doc_key(document)
        texts = _splitter(self.chunk_size, self.chunk_overlap).split_text(document.content)
        chunks = [
            Chunk(
                text=text,
                metadata={**document.metadata, "doc_id": key, "chunk_index": i},
                chunk_id=f"{key}_{i}",
            )
            for i, text in enumerate(texts)
        ]
        return chunks, []


class MarkdownChunker:
    """
    Estratégia 2: Document-aware chunking para Markdown.

    Como funciona:
      Usa a estrutura de headers (H1, H2, H3) para definir seções naturais.
      Seções longas são subdivididas com o RecursiveChunker.
      Cada chunk herda o caminho da seção ("RAG > Chunking > Overlap"),
      que também é prefixado ao texto — o chunk fica auto-explicativo.
    """

    HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200, **_):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> tuple[list[Chunk], list[Chunk]]:
        try:
            from langchain_text_splitters import MarkdownHeaderTextSplitter
        except ImportError:
            raise ImportError("pip install langchain-text-splitters")

        key = doc_key(document)
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.HEADERS, strip_headers=True)
        sub_splitter = _splitter(self.chunk_size, self.chunk_overlap)

        chunks = []
        for section in md_splitter.split_text(document.content):
            headers = [section.metadata[h] for _, h in self.HEADERS if h in section.metadata]
            section_path = " > ".join(headers)
            for text in sub_splitter.split_text(section.page_content):
                idx = len(chunks)
                chunks.append(Chunk(
                    text=f"[{section_path}]\n{text}" if section_path else text,
                    metadata={
                        **document.metadata,
                        "doc_id": key,
                        "chunk_index": idx,
                        "section": section_path,
                    },
                    chunk_id=f"{key}_{idx}",
                ))
        return chunks, []


class ParentChildChunker:
    """
    Estratégia 3: Parent-Child chunking (hierárquico).

    Como funciona:
      Indexa chunks PEQUENOS (filhos) para precisão de busca,
      mas entrega ao LLM o chunk GRANDE (pai) que contém o filho encontrado.

    Esta é geralmente a abordagem com melhor relação custo-benefício
    depois que você superou o baseline.
    """

    def __init__(
        self,
        child_chunk_size: int = 400,
        parent_chunk_size: int = 3000,
        chunk_overlap: int = 50,
        **_,
    ):
        self.child_size = child_chunk_size
        self.parent_size = parent_chunk_size
        self.overlap = min(chunk_overlap, child_chunk_size // 4)

    def split(self, document: Document) -> tuple[list[Chunk], list[Chunk]]:
        key = doc_key(document)
        parent_splitter = _splitter(self.parent_size, self.overlap)
        child_splitter = _splitter(self.child_size, self.overlap)

        parents, children = [], []
        for p_idx, p_text in enumerate(parent_splitter.split_text(document.content)):
            parent_id = f"{key}_p{p_idx}"
            parents.append(Chunk(
                text=p_text,
                metadata={**document.metadata, "doc_id": key, "chunk_type": "parent"},
                chunk_id=parent_id,
            ))
            for c_idx, c_text in enumerate(child_splitter.split_text(p_text)):
                children.append(Chunk(
                    text=c_text,
                    metadata={
                        **document.metadata,
                        "doc_id": key,
                        "chunk_type": "child",
                        "chunk_index": len(children),
                        "parent_id": parent_id,
                    },
                    chunk_id=f"{parent_id}_c{c_idx}",
                ))
        return children, parents


class AutoChunker:
    """Markdown/HTML/DOCX (que viram Markdown no loader) → MarkdownChunker; resto → RecursiveChunker."""

    STRUCTURED = {"md", "html", "htm", "docx"}

    def __init__(self, **kwargs):
        self.markdown = MarkdownChunker(**kwargs)
        self.recursive = RecursiveChunker(**kwargs)

    def split(self, document: Document) -> tuple[list[Chunk], list[Chunk]]:
        if document.metadata.get("file_type") in self.STRUCTURED:
            return self.markdown.split(document)
        return self.recursive.split(document)


def get_chunker(strategy: str = "auto", **kwargs):
    """Factory function para escolher a estratégia de chunking."""
    strategies = {
        "auto": AutoChunker,
        "recursive": RecursiveChunker,
        "markdown": MarkdownChunker,
        "parent_child": ParentChildChunker,
    }
    if strategy not in strategies:
        raise ValueError(f"Estratégia '{strategy}' desconhecida. Opções: {list(strategies)}")

    return strategies[strategy](**kwargs)
