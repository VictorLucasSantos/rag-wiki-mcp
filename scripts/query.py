"""
scripts/query.py — Consulta ao sistema RAG (fases 6-7).

Uso:
  python scripts/query.py -q "O que é RAG?"
  python scripts/query.py -q "..." -v                 # mostra os chunks recuperados
  python scripts/query.py -q "..." --rerank --multi-query 3 --hyde
  python scripts/query.py --retrieve-only -q "..."    # só a busca, sem chamar o LLM
  python scripts/query.py                             # modo interativo
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # emojis no console do Windows
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import RAGPipeline


def print_retrieval(retrieval: dict) -> None:
    if len(retrieval["queries"]) > 1:
        print("\n  🔀 Variações da pergunta:")
        for q in retrieval["queries"][1:]:
            print(f"     - {q}")
    if retrieval["hyde"]:
        print(f"\n  💭 HyDE: {retrieval['hyde'][:200]}...")
    print(f"\n  📚 {len(retrieval['docs'])} chunks (de {retrieval['candidates']} candidatos):")
    for i, doc in enumerate(retrieval["docs"], 1):
        score = doc.get("rerank_score", doc.get("rrf_score", doc.get("score", 0)))
        meta = doc.get("metadata", {})
        where = meta.get("file_name", "?") + (f" — {meta['section']}" if meta.get("section") else "")
        preview = doc["text"][:140].replace("\n", " ")
        print(f"  [{i}] {score:.4f} | {where}\n      {preview}...")


def print_sources(result: dict) -> None:
    if result["cited"]:
        print("\n📎 Fontes:")
        for s in result["cited"]:
            print(f"   [{s['n']}] {s['label']}")
    elif result["sources"]:
        print("\n📎 Nenhuma fonte citada (use -v para ver o que foi recuperado)")


def run(pipeline: RAGPipeline, question: str, verbose: bool, retrieve_only: bool) -> None:
    if retrieve_only:
        print_retrieval(pipeline.retrieve(question))
        return

    print("\n💬 ", end="")
    result = pipeline.ask(question, stream=lambda t: print(t, end="", flush=True))
    print()
    print_sources(result)
    if verbose:
        print_retrieval(result["retrieval"])


def main():
    parser = argparse.ArgumentParser(description="Consulta ao sistema RAG")
    parser.add_argument("--question", "-q", default=None, help="Pergunta (sem ela: modo interativo)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Exibe os chunks recuperados")
    parser.add_argument("--retrieve-only", action="store_true", help="Só executa a busca")
    parser.add_argument("--rerank", action="store_true", default=None, help="Re-ranking com cross-encoder")
    parser.add_argument("--multi-query", type=int, default=None, metavar="N", help="Gera N variações da pergunta")
    parser.add_argument("--hyde", action="store_true", default=None, help="Usa HyDE")
    args = parser.parse_args()

    pipeline = RAGPipeline(use_reranker=args.rerank, multi_query=args.multi_query, use_hyde=args.hyde)

    if args.question:
        run(pipeline, args.question, args.verbose, args.retrieve_only)
        return

    print("\n🤖 RAG — modo interativo (digite 'sair' para encerrar)")
    while True:
        try:
            question = input("\n❓ ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"sair", "exit", "quit", "q"}:
            break
        if question:
            run(pipeline, question, args.verbose, args.retrieve_only)


if __name__ == "__main__":
    main()
