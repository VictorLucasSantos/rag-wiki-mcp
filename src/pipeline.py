"""
src/pipeline.py — Monta os componentes a partir do config e expõe as duas fases:

  Indexer.index(path)     → fases 1-5 (carregar, limpar, chunkar, embedar, salvar)
  RAGPipeline.ask(q)      → fases 6-7 (buscar, gerar)

Indexação INCREMENTAL:
  Cada chunk guarda o content_hash do arquivo de origem. Ao reindexar,
  arquivos inalterados são pulados, alterados são substituídos e arquivos
  apagados da pasta têm seus chunks removidos.
"""

import re
from dataclasses import replace
from pathlib import Path

import config
from src.ingestion.chunker import doc_key, get_chunker
from src.ingestion.cleaner import clean_document
from src.ingestion.embedder import get_embedder
from src.ingestion.loader import DocumentLoader, source_id
from src.llm import get_llm
from src.retrieval.parent_store import ParentStore
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import get_vector_store


def build_embedder(model_name: str | None = None):
    cfg = config.embedding_config
    return get_embedder(
        backend=cfg.backend,
        model_name=model_name or cfg.model_name,
        device=cfg.device,
        batch_size=cfg.batch_size,
        api_key=config.OPENAI_API_KEY,
    )


def collection_for(model_name: str, strategy: str | None = None) -> str:
    """
    Uma coleção por combinação modelo de embedding + chunker (vetores de modelos
    diferentes não se misturam, e chunkers diferentes geram trechos diferentes).
    Os valores do config ficam com o nome original, para os índices já existentes continuarem valendo.
    """
    name = config.vector_store_config.collection_name
    if model_name != config.embedding_config.model_name:
        name += f"__{re.sub(r'[^a-zA-Z0-9]+', '-', model_name).strip('-').lower()}"
    if strategy and strategy != config.chunking_config.strategy:
        name += f"__{strategy}"
    return name


def build_vector_store(embedder, strategy: str | None = None):
    cfg = config.vector_store_config
    kwargs = {}
    if cfg.backend == "qdrant":
        kwargs["vector_size"] = len(embedder.embed_query("dimensão"))
    return get_vector_store(
        backend=cfg.backend,
        collection_name=collection_for(embedder.model_name, strategy),
        path=cfg.chroma_path,
        url=cfg.qdrant_url,
        embedding_model=embedder.model_name,
        hnsw_m=cfg.hnsw_m,
        hnsw_ef_construction=cfg.hnsw_ef_construction,
        hnsw_ef_search=cfg.hnsw_ef_search,
        **kwargs,
    )


class Indexer:
    def __init__(self, strategy: str | None = None, contextual: bool | None = None, embedder=None, vector_store=None):
        cc = config.chunking_config
        self.strategy = strategy or cc.strategy
        self.contextual = cc.contextual_retrieval if contextual is None else contextual
        self.chunker = get_chunker(
            self.strategy,
            chunk_size=cc.chunk_size,
            chunk_overlap=cc.chunk_overlap,
            child_chunk_size=cc.child_chunk_size,
            parent_chunk_size=cc.parent_chunk_size,
        )
        self.embedder = embedder or build_embedder()
        self.vector_store = vector_store or build_vector_store(self.embedder)
        self.parent_store = ParentStore(config.vector_store_config.parent_store_path)
        self._contextualizer = None

    @property
    def contextualizer(self):
        if self._contextualizer is None:
            from src.ingestion.contextualizer import Contextualizer
            lc = config.llm_config
            # Documento + instruções precisam caber na janela do modelo (~3 chars/token, com margem)
            max_doc_chars = (lc.ollama_num_ctx - 1000) * 3 if lc.backend == "ollama" else 500_000
            self._contextualizer = Contextualizer(get_llm("aux"), max_doc_chars=max_doc_chars)
        return self._contextualizer

    def _remove(self, doc_id: str) -> None:
        self.vector_store.delete_document(doc_id)
        self.parent_store.delete_document(doc_id)

    def remove_source(self, source: str) -> bool:
        """Remove do índice um documento pelo caminho ("docs/manual.pdf")."""
        info = self.vector_store.indexed_documents().get(source)
        if not info:
            return False
        self._remove(info["doc_id"])
        self.parent_store.save()
        return True

    def index(self, source_path: str, force: bool = False) -> dict:
        model = self.embedder.model_name
        indexed = self.vector_store.indexed_documents()

        # Versionamento do embedding: nunca misturar vetores de modelos diferentes
        other_models = {d["embedding_model"] for d in indexed.values()} - {model, None}
        if other_models:
            raise RuntimeError(
                f"A base foi indexada com {other_models}, mas o config usa '{model}'. "
                "Vetores de modelos diferentes são incompatíveis: rode com --rebuild."
            )

        print("\n📄 [Fase 1-2] Carregando e limpando documentos...")
        documents = [clean_document(d) for d in DocumentLoader().load(source_path)]
        stats = {"new": 0, "updated": 0, "unchanged": 0, "removed": 0, "chunks": 0}

        # Arquivos que sumiram da pasta → remove do índice
        root = Path(source_path)
        if root.is_dir():
            prefix = source_id(root).rstrip("/") + "/"
            current = {d.metadata["source"] for d in documents}
            for source, info in indexed.items():
                if source and source.startswith(prefix) and source not in current:
                    print(f"  🗑️  Removido: {source}")
                    self._remove(info["doc_id"])
                    stats["removed"] += 1

        print(f"\n✂️  [Fase 3-5] Chunking ({self.strategy}), embeddings e armazenamento...")
        for doc in documents:
            source = doc.metadata["source"]
            previous = indexed.get(source)
            if previous and previous["content_hash"] == doc.metadata["content_hash"] and not force:
                stats["unchanged"] += 1
                continue
            if previous:
                self._remove(previous["doc_id"])
                stats["updated"] += 1
            else:
                stats["new"] += 1

            doc.metadata["chunk_strategy"] = self.strategy
            chunks, parents = self.chunker.split(doc)
            if not chunks:
                continue
            if self.contextual:
                print(f"  🧩 Contextual retrieval em {len(chunks)} chunks de {doc.metadata['file_name']}...")
                chunks = self.contextualizer.contextualize(doc, chunks)

            self.vector_store.upsert(self.embedder.embed_documents(chunks))
            self.parent_store.add(parents)
            stats["chunks"] += len(chunks)
            print(f"  → {doc.metadata['file_name']}: {len(chunks)} chunks (doc_id={doc_key(doc)})")

        self.parent_store.save()
        return stats

    def rebuild(self) -> None:
        """Apaga tudo (ex.: ao trocar o modelo de embedding)."""
        for info in self.vector_store.indexed_documents().values():
            self._remove(info["doc_id"])
        self.parent_store.save()


class RAGPipeline:
    def __init__(
        self,
        use_reranker: bool | None = None,
        multi_query: int | None = None,
        use_hyde: bool | None = None,
        embedder=None,
        vector_store=None,
    ):
        self.embedder = embedder or build_embedder()
        self.vector_store = vector_store or build_vector_store(self.embedder)
        self._reranker = None
        self._query_transformer = None
        self.retriever = Retriever(self.vector_store, self.embedder, config.retrieval_config)
        self.configure(use_reranker=use_reranker, multi_query=multi_query, use_hyde=use_hyde)
        self.refresh()
        self._generator = None

    def configure(self, use_reranker: bool | None = None, multi_query: int | None = None, use_hyde: bool | None = None) -> None:
        """
        Liga/desliga etapas opcionais da busca (None = mantém o valor do config).
        Reranker e transformador de query são carregados na primeira vez que forem
        ligados e reaproveitados depois — trocar a configuração não recarrega modelos.
        """
        overrides = {"use_reranker": use_reranker, "multi_query": multi_query, "use_hyde": use_hyde}
        rc = replace(config.retrieval_config, **{k: v for k, v in overrides.items() if v is not None})

        if rc.use_reranker and self._reranker is None:
            from src.retrieval.reranker import CrossEncoderReranker
            self._reranker = CrossEncoderReranker(rc.reranker_model, device=config.embedding_config.device)
        if (rc.multi_query or rc.use_hyde) and self._query_transformer is None:
            from src.retrieval.query_transform import QueryTransformer
            self._query_transformer = QueryTransformer(get_llm("aux"))

        self.retriever.config = rc
        self.retriever.reranker = self._reranker if rc.use_reranker else None
        self.retriever.query_transformer = self._query_transformer if (rc.multi_query or rc.use_hyde) else None

    def set_embedding(self, embedder, vector_store) -> None:
        """Troca o modelo de embedding (e o índice correspondente) sem recarregar o resto."""
        self.embedder = self.retriever.embedder = embedder
        self.vector_store = self.retriever.vector_store = vector_store
        self.refresh()

    def refresh(self) -> None:
        """Recarrega BM25 e parent store após uma (re)indexação feita no mesmo processo."""
        parent_store = ParentStore(config.vector_store_config.parent_store_path)
        self.retriever.parent_store = parent_store if parent_store.parents else None
        self.retriever._bm25 = None

    @property
    def generator(self):
        if self._generator is None:
            from src.generation.generator import RAGGenerator
            lc, gc = config.llm_config, config.generation_config
            self._generator = RAGGenerator(
                get_llm("main"),
                # Modelos locais pequenos às vezes entram em loop: limite menor
                max_tokens=lc.max_tokens if lc.backend == "anthropic" else 2048,
                max_context_chars=gc.max_context_chars,
                reorder=gc.reorder_lost_in_middle,
            )
        return self._generator

    def retrieve(self, question: str, filters: dict | None = None) -> dict:
        return self.retriever.retrieve(question, filters=filters)

    def ask(self, question: str, filters: dict | None = None, stream=None) -> dict:
        """stream: callback opcional on_text(str) para exibir a resposta conforme chega."""
        retrieval = self.retrieve(question, filters)
        result = self.generator.generate(question, retrieval["docs"], on_text=stream)
        result["retrieval"] = retrieval
        return result
