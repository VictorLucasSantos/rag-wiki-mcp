"""
src/retrieval/query_transform.py — Transformações de query antes da busca.

Multi-Query:
  O LLM reescreve a pergunta de N formas diferentes. Cada variação é buscada
  e tudo é fundido com RRF. Aumenta o recall em perguntas ambíguas ou mal
  formuladas ("aquele negócio de dividir os textos" → "estratégias de chunking").

HyDE (Hypothetical Document Embeddings):
  O LLM escreve uma resposta HIPOTÉTICA à pergunta, e buscamos por documentos
  parecidos com essa resposta. Funciona porque resposta↔documento são mais
  parecidos entre si do que pergunta↔documento. Mesmo que a resposta
  hipotética tenha erros, ela usa o vocabulário certo.

Ambas custam uma chamada extra de LLM por pergunta (latência + custo).
"""

import json

from src.llm import parse_json

QUERIES_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
    "additionalProperties": False,
}


class QueryTransformer:
    def __init__(self, llm):
        self.llm = llm

    def multi_query(self, question: str, n: int = 3) -> list[str]:
        """Retorna até n reformulações da pergunta (sem incluir a original)."""
        response = self.llm.complete(
            f"Gere {n} reformulações diferentes da pergunta abaixo para melhorar a busca "
            "em uma base de documentos. Varie vocabulário, use sinônimos e termos técnicos "
            "prováveis, e decomponha a pergunta se ela tiver várias partes. Mantenha o idioma "
            f'da pergunta. Responda em JSON: {{"queries": [...]}}\n\n<pergunta>{question}</pergunta>',
            max_tokens=1024,
            schema=QUERIES_SCHEMA,
        )
        if response.refused or not response.text:
            return []
        try:
            queries = [q.strip() for q in parse_json(response)["queries"] if q.strip()]
        except (json.JSONDecodeError, KeyError, TypeError):
            return []  # modelo local devolveu algo fora do formato: segue só com a original
        return [q for q in queries if q != question][:n]

    def hyde(self, question: str) -> str:
        """Gera um parágrafo hipotético que responderia à pergunta."""
        response = self.llm.complete(
            "Escreva um parágrafo curto, no estilo de um trecho de documentação, que responda "
            "à pergunta abaixo. Não diga que é hipotético, apenas escreva o trecho.\n\n"
            f"<pergunta>{question}</pergunta>",
            max_tokens=400,
        )
        return "" if response.refused else response.text
