"""
src/ingestion/contextualizer.py — Contextual Retrieval (opcional, na indexação).

Problema:
  Um chunk como "A receita cresceu 3% no trimestre" perdeu o contexto
  (qual empresa? qual trimestre?). Nem o embedding nem o BM25 conseguem
  recuperá-lo para a pergunta "Quanto a ACME cresceu no 2º tri de 2023?".

Solução (Anthropic, 2024):
  Para cada chunk, um LLM lê o documento inteiro e escreve 1-2 frases
  situando o chunk. Esse snippet é prefixado ao chunk SÓ para indexação
  (embedding + BM25). Reportado: -49% de falhas de recuperação.

Custo:
  Uma chamada por chunk. Com Claude, o documento vai no início do prompt com
  prompt caching (a partir do 2º chunk sai ~90% mais barato). Com Ollama é
  grátis, mas lento, e o documento precisa caber no num_ctx.
"""

from src.ingestion.chunker import Chunk
from src.ingestion.loader import Document

PROMPT = """Aqui está o chunk que queremos situar dentro do documento acima:
<chunk>
{chunk}
</chunk>

Escreva um contexto curto (1-2 frases, no idioma do documento) que situe este chunk dentro do documento \
como um todo, para melhorar a recuperação do chunk em buscas. Mencione entidades, datas e o tema da seção \
quando ajudarem. Responda apenas com o contexto, sem preâmbulo."""


class Contextualizer:
    def __init__(self, llm, max_doc_chars: int = 500_000):
        self.llm = llm
        # Documento maior que isso não cabe na janela do modelo: pula em vez de cortar
        self.max_doc_chars = max_doc_chars

    def contextualize(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
        if len(document.content) > self.max_doc_chars:
            print(f"  ⚠️  {document.metadata.get('file_name')} é grande demais para "
                  f"contextual retrieval ({len(document.content)} chars) — pulando.")
            return chunks

        prefix = f"<documento>\n{document.content}\n</documento>"
        for chunk in chunks:
            try:
                response = self.llm.complete(PROMPT.format(chunk=chunk.text), prefix=prefix, max_tokens=300)
            except Exception as e:  # um chunk sem contexto não deve derrubar a indexação
                print(f"  ⚠️  Falha ao contextualizar {chunk.chunk_id}: {e}")
                continue
            if not response.refused:
                chunk.context = response.text
        return chunks
