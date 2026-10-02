"""
src/evaluation/metrics.py — Avaliação do sistema RAG.

"Funciona na minha pergunta de teste" não é avaliação. Precisamos de um
golden set (perguntas + fontes esperadas + resposta de referência) e métricas
separadas para as duas metades do sistema:

RETRIEVAL (barato, determinístico — rode a cada mudança de chunking/embedding):
  - Hit rate@k : % de perguntas em que alguma fonte esperada aparece no top-k
  - MRR        : média de 1/posição da primeira fonte correta (premia vir no topo)
  - Recall@k   : % das fontes esperadas que apareceram no top-k

GERAÇÃO (LLM-as-judge, custa chamadas de API — inspirado no RAGAS):
  - Faithfulness     : fração das afirmações da resposta suportadas pelo contexto
  - Answer relevancy : a resposta responde à pergunta? (1-5)
  - Correctness      : concorda com a resposta de referência? (1-5)
"""

import json

from src.llm import parse_json


def _matches_source(doc: dict, expected: str) -> bool:
    meta = doc.get("metadata", {})
    return expected in {meta.get("file_name"), meta.get("source")}


def retrieval_metrics(docs: list[dict], expected_sources: list[str]) -> dict:
    """Métricas de uma pergunta. `docs` em ordem de ranking."""
    if not expected_sources:
        return {}
    first_hit = next(
        (rank for rank, d in enumerate(docs, 1) if any(_matches_source(d, e) for e in expected_sources)),
        None,
    )
    found = {e for e in expected_sources if any(_matches_source(d, e) for d in docs)}
    return {
        "hit": first_hit is not None,
        "reciprocal_rank": 1 / first_hit if first_hit else 0.0,
        "recall": len(found) / len(expected_sources),
    }


def aggregate(rows: list[dict]) -> dict:
    """Média de cada métrica numérica/booleana sobre todas as perguntas."""
    keys = {k for r in rows for k, v in r.items() if isinstance(v, (int, float, bool))}
    summary = {}
    for k in sorted(keys):
        values = [float(r[k]) for r in rows if k in r]
        summary[k] = sum(values) / len(values) if values else 0.0
    return summary


JUDGE_PROMPT = """Avalie a resposta de um sistema RAG.

<contexto>
{context}
</contexto>

<pergunta>
{question}
</pergunta>

<resposta>
{answer}
</resposta>

<resposta_de_referencia>
{reference}
</resposta_de_referencia>

Critérios:
- claims_total: quantas afirmações factuais distintas a resposta faz (dizer "não encontrei" conta 0).
- claims_supported: quantas dessas afirmações são sustentadas pelo CONTEXTO (não pelo seu conhecimento).
- unsupported_claims: liste as afirmações não sustentadas pelo contexto.
- answer_relevancy: 1-5, o quanto a resposta atende à pergunta.
- correctness: 1-5, o quanto concorda com a resposta de referência (use 0 se não houver referência).

Responda em JSON com os campos claims_total, claims_supported, unsupported_claims, answer_relevancy e correctness."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims_total": {"type": "integer"},
        "claims_supported": {"type": "integer"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "answer_relevancy": {"type": "integer"},
        "correctness": {"type": "integer"},
    },
    "required": ["claims_total", "claims_supported", "unsupported_claims", "answer_relevancy", "correctness"],
    "additionalProperties": False,
}


class LLMJudge:
    def __init__(self, llm):
        self.llm = llm

    def judge(self, question: str, answer: str, docs: list[dict], reference: str = "") -> dict:
        context = "\n\n---\n\n".join(d["text"] for d in docs) or "(nenhum documento recuperado)"
        response = self.llm.complete(
            JUDGE_PROMPT.format(
                context=context, question=question, answer=answer, reference=reference or "(sem referência)",
            ),
            max_tokens=4000,
            schema=JUDGE_SCHEMA,
        )
        if response.refused:
            return {"judge_error": "refusal"}
        try:
            verdict = parse_json(response)
        except json.JSONDecodeError:
            return {"judge_error": "resposta fora do formato JSON"}

        total = verdict["claims_total"]
        result = {
            # Sem afirmações (ex.: "não encontrei") não há o que alucinar
            "faithfulness": min(verdict["claims_supported"] / total, 1.0) if total else 1.0,
            "answer_relevancy": verdict["answer_relevancy"],
            "unsupported_claims": verdict["unsupported_claims"],
        }
        if reference:
            result["correctness"] = verdict["correctness"]
        return result
