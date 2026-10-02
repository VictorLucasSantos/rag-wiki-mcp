from src.ingestion.chunker import AutoChunker, MarkdownChunker, ParentChildChunker, RecursiveChunker, get_chunker
from src.ingestion.cleaner import clean_text
from src.ingestion.loader import Document

LONG_TEXT = "\n\n".join(f"Parágrafo {i}. " + "Texto de exemplo sobre RAG. " * 20 for i in range(10))

MARKDOWN = """# Política

## Férias

Todo colaborador tem 30 dias de férias.

## Ausências

### Day off

Um day off no mês do aniversário.
"""


def doc(content, file_type="txt", source="docs/a.txt"):
    return Document(content=content, metadata={"source": source, "file_name": source.split("/")[-1], "file_type": file_type})


def test_recursive_respects_size_and_ids_are_deterministic():
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    chunks, parents = chunker.split(doc(LONG_TEXT))
    assert parents == []
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)
    again, _ = chunker.split(doc(LONG_TEXT))
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in again]
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_ids_differ_between_documents():
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    a, _ = chunker.split(doc(LONG_TEXT, source="docs/a.txt"))
    b, _ = chunker.split(doc(LONG_TEXT, source="docs/b.txt"))
    assert a[0].chunk_id != b[0].chunk_id


def test_markdown_keeps_section_path():
    chunks, _ = MarkdownChunker(chunk_size=500, chunk_overlap=0).split(doc(MARKDOWN, "md", "docs/p.md"))
    day_off = next(c for c in chunks if "day off" in c.text.lower())
    assert day_off.metadata["section"] == "Política > Ausências > Day off"
    assert day_off.text.startswith("[Política > Ausências > Day off]")


def test_parent_child_links_children_to_parents():
    children, parents = ParentChildChunker(child_chunk_size=200, parent_chunk_size=1000).split(doc(LONG_TEXT))
    parent_ids = {p.chunk_id for p in parents}
    assert len(children) > len(parents) > 1
    assert all(c.metadata["parent_id"] in parent_ids for c in children)
    for c in children:
        parent = next(p for p in parents if p.chunk_id == c.metadata["parent_id"])
        assert c.text in parent.text


def test_auto_picks_strategy_by_file_type():
    auto = AutoChunker(chunk_size=500, chunk_overlap=0)
    md_chunks, _ = auto.split(doc(MARKDOWN, "md", "docs/p.md"))
    assert "section" in md_chunks[0].metadata
    txt_chunks, _ = auto.split(doc(MARKDOWN, "txt"))
    assert "section" not in txt_chunks[0].metadata


def test_indexed_text_includes_context():
    chunks, _ = get_chunker("recursive", chunk_size=500, chunk_overlap=0).split(doc("Receita cresceu 3%."))
    chunks[0].context = "Relatório da ACME, 2º trimestre de 2023."
    assert chunks[0].indexed_text.startswith("Relatório da ACME")
    assert chunks[0].text == "Receita cresceu 3%."


def test_cleaner_removes_repeated_footer_and_hyphenation():
    pages = "\n\n".join(
        f"[Página {i}]\nConteúdo da página {i} com informa-\nção importante.\nEmpresa X — Confidencial\n{i}"
        for i in range(1, 5)
    )
    cleaned = clean_text(pages)
    assert "Confidencial" not in cleaned
    assert "informação" in cleaned
    assert "\n4\n" not in cleaned + "\n"
