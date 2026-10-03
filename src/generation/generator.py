"""
src/generation/generator.py — Fase 7b: Geração da resposta.

Recebe a pergunta + chunks recuperados e gera uma resposta fundamentada
e com citações. A montagem do prompt fica em prompt_builder.py e o acesso
ao modelo (Claude ou Ollama) fica em src/llm.py.

Decisão: sem documentos recuperados → não chama o LLM; responde que não
encontrou. (Cair para "conhecimento geral" esconde falhas de retrieval e alucina.)
"""

import re

from src.generation.prompt_builder import build_messages

NOT_FOUND = "Não encontrei essa informação nos documentos disponíveis."


def cited_sources(answer: str, sources: list[dict]) -> list[dict]:
    """Filtra as fontes que a resposta realmente citou com [n]."""
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return [s for s in sources if s["n"] in cited]


class RAGGenerator:
    """
    Exemplo de uso:
        generator = RAGGenerator(get_llm("main"))
        result = generator.generate("O que é chunking?", docs)
        print(result["answer"], result["sources"])
    """

    def __init__(
        self,
        llm,
        max_tokens: int = 16000,
        max_context_chars: int = 12000,
        reorder: bool = True,
        max_history_chars: int = 3000,
    ):
        self.llm = llm
        self.max_tokens = max_tokens
        self.max_context_chars = max_context_chars
        self.reorder = reorder
        self.max_history_chars = max_history_chars

    def generate(self, question: str, docs: list[dict], on_text=None, history: list[dict] | None = None) -> dict:
        """
        Retorna {"answer", "sources", "cited", "num_docs", "model", "usage", "query"}.
        on_text: callback opcional de streaming.
        history: trocas anteriores da conversa (ver prompt_builder.format_history).
        """
        if not docs:
            if on_text:
                on_text(NOT_FOUND)
            return self._result(question, NOT_FOUND, [], docs)

        system, user, sources = build_messages(
            question, docs, self.max_context_chars, self.reorder,
            history=history, max_history_chars=self.max_history_chars,
        )
        response = self.llm.complete(user, system=system, max_tokens=self.max_tokens, on_text=on_text)

        if response.refused:
            answer = "O modelo recusou responder a esta pergunta."
        else:
            answer = response.text
            if response.truncated:
                answer += "\n\n[resposta truncada: limite de tokens atingido]"
        return self._result(question, answer, sources, docs, response)

    def _result(self, question: str, answer: str, sources: list[dict], docs: list[dict], response=None) -> dict:
        return {
            "query": question,
            "answer": answer,
            "sources": sources,                       # todas as fontes do prompt, numeradas
            "cited": cited_sources(answer, sources),  # só as citadas na resposta
            "num_docs": len(docs),
            "model": response.model if response else None,
            "usage": response.usage if response else None,
        }
