"""
src/retrieval/retriever.py — Fase 6 completa: orquestra todas as técnicas de busca.

Pipeline (cada etapa entre [] é opcional, via config):

  pergunta
    ├─ [multi-query]  → N reformulações
    ├─ [HyDE]         → documento hipotético (só busca densa)
    ▼
  para cada variação: busca densa (vector store) + BM25
    ▼
  fusão RRF ponderada  → k_initial candidatos
    ▼
  [re-ranking com cross-encoder usando a pergunta ORIGINAL]
    ▼
  [parent-child: troca filhos pelos pais e deduplica]
    ▼
  top k_final → prompt
"""

from src.retrieval.hybrid_search import BM25Retriever, reciprocal_rank_fusion


class Retriever:
    def __init__(
        self,
        vector_store,
        embedder,
        config,
        reranker=None,
        query_transformer=None,
        parent_store=None,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.config = config
        self.reranker = reranker
        self.query_transformer = query_transformer
        self.parent_store = parent_store
        self._bm25: BM25Retriever | None = None

    @property
    def bm25(self) -> BM25Retriever:
        """Índice BM25 construído a partir do que está no vector store (lazy)."""
        if self._bm25 is None:
            self._bm25 = BM25Retriever()
            self._bm25.index(self.vector_store.get_all())
        return self._bm25

    def expand_queries(self, question: str) -> tuple[list[str], str]:
        """Retorna (queries para busca densa+BM25, documento HyDE ou "")."""
        queries = [question]
        hyde_doc = ""
        if self.query_transformer:
            if self.config.multi_query:
                queries += self.query_transformer.multi_query(question, n=self.config.multi_query)
            if self.config.use_hyde:
                hyde_doc = self.query_transformer.hyde(question)
        return queries, hyde_doc

    def retrieve(self, question: str, filters: dict | None = None) -> dict:
        """
        Retorna {"docs": [...], "queries": [...], "hyde": str, "candidates": int}.
        Os metadados extras ajudam a depurar por que um chunk veio (ou não).
        """
        cfg = self.config
        queries, hyde_doc = self.expand_queries(question)

        result_lists, weights = [], []
        for q in queries:
            result_lists.append(self.vector_store.search(self.embedder.embed_query(q), k=cfg.k_initial, filters=filters))
            weights.append(cfg.semantic_weight)
            if cfg.use_hybrid:
                result_lists.append(self.bm25.search(q, k=cfg.k_initial, filters=filters))
                weights.append(cfg.keyword_weight)
        if hyde_doc:
            # O documento hipotético é um "documento", então usa o prefixo de documento
            hyde_vec = self.embedder.embed_texts([hyde_doc], is_query=False)[0]
            result_lists.append(self.vector_store.search(hyde_vec, k=cfg.k_initial, filters=filters))
            weights.append(cfg.semantic_weight)

        candidates = reciprocal_rank_fusion(result_lists, k=cfg.rrf_k, weights=weights)[:cfg.k_initial]

        if self.reranker:
            # Pega mais que k_final: a expansão parent-child pode deduplicar
            docs = self.reranker.rerank(
                question, candidates, top_k=cfg.k_final * 3, min_score=cfg.min_rerank_score,
            )
        else:
            docs = candidates

        if self.parent_store:
            docs = self.parent_store.expand(docs)

        return {
            "docs": docs[:cfg.k_final],
            "queries": queries,
            "hyde": hyde_doc,
            "candidates": len(candidates),
        }
