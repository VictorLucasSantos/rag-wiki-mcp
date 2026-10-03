from types import SimpleNamespace

import config
from config import RetrievalConfig
from src.evaluation.metrics import aggregate, retrieval_metrics
from src.ingestion.chunker import Chunk
from src.pipeline import collection_for
from src.retrieval.hybrid_search import BM25Retriever, reciprocal_rank_fusion, tokenize
from src.retrieval.parent_store import ParentStore
from src.retrieval.retriever import Retriever

CORPUS = [
    {"id": "a", "text": "O chamado TI-1180 bloqueia o acesso durante as férias.", "metadata": {"file_name": "ti.md"}},
    {"id": "b", "text": "Férias de 30 dias corridos por período aquisitivo.", "metadata": {"file_name": "rh.md"}},
    {"id": "c", "text": "Busca híbrida combina BM25 com vetores.", "metadata": {"file_name": "rag.md"}},
]


def test_tokenize_strips_accents_stopwords_and_keeps_identifiers():
    assert tokenize("As Férias do GPT-4 e o TI-1180") == ["ferias", "gpt-4", "ti-1180"]


def test_rrf_rewards_documents_in_both_lists():
    dense = [{"id": "x", "text": "x"}, {"id": "y", "text": "y"}]
    sparse = [{"id": "y", "text": "y"}, {"id": "z", "text": "z"}]
    fused = reciprocal_rank_fusion([dense, sparse])
    assert fused[0]["id"] == "y"
    assert abs(fused[0]["rrf_score"] - (1 / 62 + 1 / 61)) < 1e-9


def test_weighted_rrf():
    fused = reciprocal_rank_fusion([[{"id": "x", "text": ""}], [{"id": "y", "text": ""}]], weights=[0.2, 0.8])
    assert fused[0]["id"] == "y"


def test_bm25_exact_term_and_filters():
    bm25 = BM25Retriever()
    bm25.index(CORPUS)
    assert bm25.search("ti-1180")[0]["id"] == "a"
    assert [r["id"] for r in bm25.search("férias", filters={"file_name": "rh.md"})] == ["b"]
    assert bm25.search("de o a") == []


class FakeStore:
    def __init__(self, items):
        self.items = items

    def search(self, query_embedding, k=5, filters=None):
        return [dict(i, score=1.0) for i in self.items][:k]

    def get_all(self):
        return self.items


class FakeEmbedder:
    def embed_query(self, q):
        return [0.0]

    def embed_texts(self, texts, is_query=False):
        return [[0.0] for _ in texts]


def test_retriever_fuses_dense_and_bm25():
    store = FakeStore([CORPUS[2], CORPUS[1], CORPUS[0]])  # denso coloca "a" por último
    retriever = Retriever(store, FakeEmbedder(), RetrievalConfig(k_final=1, multi_query=0, use_reranker=False))
    result = retriever.retrieve("chamado TI-1180")
    assert result["docs"][0]["id"] == "a"  # BM25 puxou o termo exato para o topo


def test_retriever_uses_query_variations():
    transformer = SimpleNamespace(multi_query=lambda q, n: ["variação 1", "variação 2"], hyde=lambda q: "")
    retriever = Retriever(FakeStore(CORPUS), FakeEmbedder(), RetrievalConfig(multi_query=2), query_transformer=transformer)
    assert retriever.retrieve("pergunta")["queries"] == ["pergunta", "variação 1", "variação 2"]


def test_parent_store_expands_and_dedupes(tmp_path):
    store = ParentStore(str(tmp_path / "parents.json"))
    store.add([Chunk(text="texto pai completo", metadata={"doc_id": "d"}, chunk_id="d_p0")])
    store.save()
    reloaded = ParentStore(str(tmp_path / "parents.json"))
    docs = [
        {"id": "d_p0_c0", "text": "filho 1", "metadata": {"parent_id": "d_p0"}},
        {"id": "d_p0_c1", "text": "filho 2", "metadata": {"parent_id": "d_p0"}},
        {"id": "solto", "text": "sem pai", "metadata": {}},
    ]
    expanded = reloaded.expand(docs)
    assert [d["id"] for d in expanded] == ["d_p0", "solto"]
    assert expanded[0]["text"] == "texto pai completo"
    reloaded.delete_document("d")
    assert reloaded.parents == {}


def test_retrieval_metrics():
    docs = [{"metadata": {"file_name": "x.md"}}, {"metadata": {"file_name": "rh.md"}}]
    m = retrieval_metrics(docs, ["rh.md"])
    assert m == {"hit": True, "reciprocal_rank": 0.5, "recall": 1.0}
    assert retrieval_metrics(docs, ["zzz.md"])["hit"] is False
    assert aggregate([m, {"hit": False, "reciprocal_rank": 0.0, "recall": 0.0}])["hit"] == 0.5


def test_collection_for_keeps_default_name_and_separates_combinations():
    base = config.vector_store_config.collection_name
    model, strategy = config.embedding_config.model_name, config.chunking_config.strategy
    assert collection_for(model) == collection_for(model, strategy) == base
    names = {collection_for(m, s) for m in (model, "intfloat/multilingual-e5-small") for s in (strategy, "parent_child")}
    assert len(names) == 4
