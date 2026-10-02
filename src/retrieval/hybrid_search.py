"""
src/retrieval/hybrid_search.py — Fase 6: Busca por keyword (BM25) e fusão RRF.

Por que busca híbrida?

  Busca semântica (dense):
    ✅ Encontra sinônimos e conceitos relacionados
    ❌ Má precisão para termos exatos, nomes próprios, IDs técnicos

  Busca por keyword BM25 (sparse):
    ✅ Alta precisão para termos exatos ("CVE-2024-1234", "art. 5º")
    ❌ Não entende semântica ("NLP" ≠ "processamento de linguagem natural")

  Busca híbrida:
    ✅ Melhor dos dois mundos — tipicamente supera ambas isoladas

Combinação via RRF (Reciprocal Rank Fusion):
  Em vez de combinar scores (escalas diferentes), combina RANKS.
  score_rrf = Σ peso / (k + rank)  onde k=60 é o valor do paper original.
"""

import re
import unicodedata

# Stopwords mínimas de PT/EN: palavras que não ajudam o BM25 a discriminar
STOPWORDS = set("""
a o as os um uma uns umas de do da dos das em no na nos nas por pelo pela pelos pelas
para pra com sem sob sobre entre e ou mas que se ao aos à às é ser são foi era como
mais menos muito já não sim isso isto esse essa este esta aquele aquela seu sua seus suas
meu minha qual quais quando onde quem porque também até lhe eles elas ele ela eu você
the an of to in on for with and or is are was were be by at as it this that from
""".split())

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """
    Normaliza para BM25: minúsculas, remove acentos, mantém identificadores
    como "gpt-4" ou "v1.2" inteiros, descarta stopwords.
    """
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in _TOKEN.findall(text) if t not in STOPWORDS]


def reciprocal_rank_fusion(
    results_lists: list[list[dict]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[dict]:
    """
    Combina múltiplas listas de resultados rankeados usando RRF (ponderado).

    Por que RRF em vez de interpolação linear de scores?
      Scores de busca semântica (0-1) e BM25 (0-∞) têm escalas diferentes.
      RRF usa só a posição, tornando a fusão robusta.

    Exemplo:
      reciprocal_rank_fusion([semantic_results, bm25_results], weights=[0.6, 0.4])
      Documento no rank 1 do semântico + rank 3 do BM25 terá score alto.
    """
    weights = weights or [1.0] * len(results_lists)
    scores: dict[str, dict] = {}

    for results, weight in zip(results_lists, weights):
        for rank, result in enumerate(results):
            doc_id = result.get("id") or result["text"]
            entry = scores.setdefault(doc_id, {"rrf_score": 0.0, "doc": result})
            # rank=0 → 1/61 ≈ 0.0164 ; rank=9 → 1/70 ≈ 0.0143
            entry["rrf_score"] += weight / (k + rank + 1)

    fused = []
    for item in sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True):
        doc = dict(item["doc"])
        doc["rrf_score"] = item["rrf_score"]
        fused.append(doc)
    return fused


def _matches(metadata: dict, filters: dict | None) -> bool:
    return not filters or all(metadata.get(k) == v for k, v in filters.items())


class BM25Retriever:
    """
    Busca por keyword usando BM25 (Best Match 25).

    Calcula score baseado em:
      - TF (Term Frequency): frequência do termo no chunk
      - IDF (Inverse Document Frequency): raridade do termo no corpus
      - Normalização pelo tamanho do chunk

    Limitação: o índice fica em memória. Para bases muito grandes,
    use Elasticsearch/OpenSearch ou o modo híbrido nativo do Qdrant/Weaviate.
    """

    def __init__(self):
        self._bm25 = None
        self._corpus: list[dict] = []

    def index(self, documents: list[dict]) -> None:
        """documents: lista de dicts com "id", "text", "metadata"."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("pip install rank-bm25")

        self._corpus = documents
        if not documents:
            self._bm25 = None
            return
        # Indexa também o snippet do Contextual Retrieval, se houver
        tokenized = [tokenize(f"{d['metadata'].get('context', '')} {d['text']}") for d in documents]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 10, filters: dict | None = None) -> list[dict]:
        if self._bm25 is None:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for i in ranked:
            if scores[i] <= 0 or len(results) >= k:
                break
            if _matches(self._corpus[i]["metadata"], filters):
                results.append({**self._corpus[i], "score": float(scores[i])})
        return results
