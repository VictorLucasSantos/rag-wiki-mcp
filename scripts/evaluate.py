"""
scripts/evaluate.py — Avalia o RAG contra um golden set.

Formato do golden set (JSON):
  [
    {
      "question": "Quantos dias de férias...?",
      "expected_sources": ["politica_ferias.md"],      # file_name ou source
      "reference_answer": "30 dias corridos..."          # opcional
    }
  ]

Uso:
  python scripts/evaluate.py                         # só retrieval (rápido, sem custo de API)
  python scripts/evaluate.py --rerank                # compara configurações de busca
  python scripts/evaluate.py --generate              # + geração e LLM-as-judge (chama a API)

Os resultados detalhados são salvos em eval/results/ para comparar execuções.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")  # emojis no console do Windows
sys.path.insert(0, str(ROOT))

import config
from src.evaluation.metrics import LLMJudge, aggregate, retrieval_metrics
from src.llm import get_llm
from src.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Avalia o sistema RAG")
    parser.add_argument("--golden", default=str(ROOT / "eval" / "golden_set.json"))
    parser.add_argument("--generate", action="store_true", help="Também gera respostas e avalia com LLM-as-judge")
    parser.add_argument("--rerank", action="store_true", default=None)
    parser.add_argument("--multi-query", type=int, default=None, metavar="N")
    parser.add_argument("--hyde", action="store_true", default=None)
    args = parser.parse_args()

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    pipeline = RAGPipeline(use_reranker=args.rerank, multi_query=args.multi_query, use_hyde=args.hyde)
    judge = LLMJudge(get_llm("judge")) if args.generate else None

    rows = []
    for i, item in enumerate(golden, 1):
        question = item["question"]
        start = time.perf_counter()
        retrieval = pipeline.retrieve(question)
        row = {
            "question": question,
            "retrieved": [d.get("metadata", {}).get("file_name") for d in retrieval["docs"]],
            "retrieval_latency_s": time.perf_counter() - start,
            **retrieval_metrics(retrieval["docs"], item.get("expected_sources", [])),
        }

        if judge:
            result = pipeline.generator.generate(question, retrieval["docs"])
            row["answer"] = result["answer"]
            row.update(judge.judge(question, result["answer"], retrieval["docs"], item.get("reference_answer", "")))

        # Perguntas sem fonte esperada testam o "não encontrei" — não entram no hit rate
        status = "➖" if "hit" not in row else ("✅" if row["hit"] else "❌")
        print(f"{status} [{i}/{len(golden)}] {question[:70]}")
        if judge and row.get("unsupported_claims"):
            print(f"     ⚠️  não suportado: {row['unsupported_claims']}")
        rows.append(row)

    summary = aggregate(rows)
    print("\n" + "=" * 60)
    print(f"📊 Resumo ({len(rows)} perguntas, k={config.retrieval_config.k_final})")
    for key, value in summary.items():
        print(f"   {key:22s} {value:.3f}")
    print("=" * 60)

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "config": {
            "embedding": config.embedding_config.model_name,
            "chunking": config.chunking_config.strategy,
            "retrieval": vars(pipeline.retriever.config),
        },
        "summary": summary,
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 Detalhes em {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
