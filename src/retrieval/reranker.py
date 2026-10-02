"""
src/retrieval/reranker.py — Re-ranking com cross-encoder (segunda camada de relevância).

Bi-encoder (embedding) × Cross-encoder:
  O embedding codifica query e documento SEPARADAMENTE — rápido, mas raso.
  O cross-encoder lê query + documento JUNTOS e julga a relevância — muito
  mais preciso, porém caro demais para rodar no corpus inteiro.

Pipeline clássico:
  busca híbrida (k=30 candidatos) → cross-encoder → top 5 para o LLM

Modelo padrão: BAAI/bge-reranker-v2-m3 (multilíngue, bom em PT-BR, open source).
Alternativa gerenciada: Cohere Rerank API.
"""


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError("pip install sentence-transformers")
            print(f"  🎯 Carregando reranker {self.model_name}...")
            self._model = CrossEncoder(self.model_name, device=self.device, max_length=512)
        return self._model

    def rerank(self, query: str, docs: list[dict], top_k: int = 5, min_score: float | None = None) -> list[dict]:
        if not docs:
            return []
        scores = self.model.predict([(query, d["text"]) for d in docs])
        reranked = sorted(
            ({**d, "rerank_score": float(s)} for d, s in zip(docs, scores)),
            key=lambda d: d["rerank_score"],
            reverse=True,
        )
        if min_score is not None:
            reranked = [d for d in reranked if d["rerank_score"] >= min_score]
        return reranked[:top_k]
