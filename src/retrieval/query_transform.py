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

Condensação (perguntas de acompanhamento numa conversa):
  "E posso vender parte delas?" não tem o que buscar sozinha. O LLM reescreve
  a pergunta como independente usando o histórico ("Posso vender parte das
  minhas férias?") e é ESSA versão que vai para a busca.

Todas custam uma chamada extra de LLM por pergunta (latência + custo).
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

    def condense(self, question: str, history_block: str) -> str:
        """
        Reescreve uma pergunta de acompanhamento como pergunta independente.
        history_block: saída de prompt_builder.format_history. Em qualquer falha,
        devolve a pergunta original (melhor buscar por ela do que por nada).
        """
        if not history_block:
            return question
        response = self.llm.complete(
            f"{history_block}<pergunta>{question}</pergunta>\n\n"
            "Reescreva a pergunta acima como uma pergunta independente, que faça sentido sem "
            "o histórico: troque pronomes e referências (\"isso\", \"ela\", \"o segundo\") pelo "
            "que eles significam na conversa. Se ela já for independente, devolva-a igual. "
            "Não responda a pergunta. Escreva apenas a pergunta reescrita, no mesmo idioma.",
            max_tokens=200,
        )
        text = "" if response.refused else response.text.strip().strip('"“”').strip()
        text = text.splitlines()[0].strip() if text else ""
        # Resposta vazia ou longa demais = o modelo não seguiu a instrução
        return text if text and len(text) <= max(300, 3 * len(question)) else question

    def hyde(self, question: str) -> str:
        """Gera um parágrafo hipotético que responderia à pergunta."""
        response = self.llm.complete(
            "Escreva um parágrafo curto, no estilo de um trecho de documentação, que responda "
            "à pergunta abaixo. Não diga que é hipotético, apenas escreva o trecho.\n\n"
            f"<pergunta>{question}</pergunta>",
            max_tokens=400,
        )
        return "" if response.refused else response.text
