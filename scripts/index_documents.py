"""
scripts/index_documents.py — Indexação da base de conhecimento (fases 1-5).

  1. Carrega documentos (loader)
  2. Limpa o texto (cleaner)
  3. Divide em chunks (chunker) [+ contextual retrieval opcional]
  4. Gera embeddings (embedder)
  5. Salva no vector store (incremental: só o que mudou)

Uso:
  python scripts/index_documents.py --source docs
  python scripts/index_documents.py --source docs --chunker parent_child --rebuild
  python scripts/index_documents.py --source docs --contextual
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # emojis no console do Windows
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.pipeline import Indexer


def main():
    parser = argparse.ArgumentParser(description="Indexa documentos para o sistema RAG")
    parser.add_argument("--source", default="docs", help="Caminho da pasta ou arquivo")
    parser.add_argument("--chunker", default=None, help="auto | recursive | markdown | parent_child")
    parser.add_argument("--contextual", action="store_true", help="Ativa Contextual Retrieval (usa a API da Anthropic)")
    parser.add_argument("--force", action="store_true", help="Reindexa mesmo arquivos inalterados")
    parser.add_argument("--rebuild", action="store_true", help="Apaga o índice inteiro antes (ex.: trocou de modelo/chunker)")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 RAG Pipeline — Fase de Indexação")
    print("=" * 60)

    indexer = Indexer(strategy=args.chunker, contextual=args.contextual or None)
    if args.rebuild:
        print("\n🧨 Apagando índice existente...")
        indexer.rebuild()

    stats = indexer.index(args.source, force=args.force)

    print("\n" + "=" * 60)
    print("✅ Indexação concluída!")
    print(f"   Novos: {stats['new']} | Atualizados: {stats['updated']} | "
          f"Inalterados: {stats['unchanged']} | Removidos: {stats['removed']}")
    print(f"   Chunks gerados: {stats['chunks']}")
    print(f"   Estratégia:     {indexer.strategy}{' + contextual' if indexer.contextual else ''}")
    print(f"   Embedding:      {config.embedding_config.model_name}")
    print(f"   Store:          {config.vector_store_config.backend}")
    print("=" * 60)


if __name__ == "__main__":
    main()
