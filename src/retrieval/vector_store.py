"""
src/retrieval/vector_store.py — Fase 5+6: Armazenamento e busca vetorial.

Responsabilidade:
  Persistir os vetores + metadados, e executar buscas de similaridade.

COMO FUNCIONA A BUSCA VETORIAL:
  Vector databases usam algoritmos ANN (Approximate Nearest Neighbor)
  porque busca exata (KNN) é cara em alta dimensão com muitos vetores.

  O algoritmo mais comum é HNSW (Hierarchical Navigable Small World):
  - Constrói um grafo onde cada ponto conecta aos seus vizinhos mais próximos
  - Navega do nível mais alto (poucos nós) ao mais baixo (busca precisa)
  - Trade-off: ef_search maior = mais preciso, mais lento

BACKENDS disponíveis:
  Chroma  → desenvolvimento local, sem configuração extra
  Qdrant  → produção, self-hosted, melhor performance e filtros

Interface comum:
  upsert(embedded_chunks)          search(query_embedding, k, filters)
  get_all()  (para o BM25)         delete_document(doc_id)
  indexed_documents()  (source → content_hash, para indexação incremental)
"""

import uuid

from src.ingestion.embedder import EmbeddedChunk


def _sanitize_metadata(metadata: dict) -> dict:
    """Vector stores só aceitam escalares: descarta None e converte o resto para str."""
    clean = {}
    for key, value in metadata.items():
        if value is None:
            continue
        clean[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return clean


def _payload(ec: EmbeddedChunk, embedding_model: str) -> dict:
    return _sanitize_metadata({
        **ec.metadata,
        "chunk_id": ec.chunk_id,
        "context": ec.chunk.context,
        # Versionamento: qual modelo gerou este vetor
        "embedding_model": embedding_model,
    })


class ChromaVectorStore:
    """
    Vector store usando Chroma — perfeito para desenvolvimento local.

    Armazena tudo em disco na pasta configurada.
    Sem necessidade de Docker ou serviços externos.
    """

    def __init__(
        self,
        collection_name: str = "rag_kb",
        path: str = "./data/chroma_db",
        embedding_model: str = "",
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 128,
        hnsw_ef_search: int = 64,
        **_,
    ):
        self.collection_name = collection_name
        self.path = path
        self.embedding_model = embedding_model
        self.hnsw = {"M": hnsw_m, "construction_ef": hnsw_ef_construction, "search_ef": hnsw_ef_search}
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            try:
                import chromadb
            except ImportError:
                raise ImportError("pip install chromadb")
            client = chromadb.PersistentClient(path=self.path)
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                configuration={"hnsw": {"space": "cosine", "max_neighbors": self.hnsw["M"],
                                        "ef_construction": self.hnsw["construction_ef"],
                                        "ef_search": self.hnsw["search_ef"]}},
            )
        return self._collection

    def upsert(self, embedded_chunks: list[EmbeddedChunk], batch_size: int = 1000) -> None:
        """
        Insere ou atualiza chunks no vector store.
        upsert = INSERT or UPDATE (por chunk_id) → re-indexar não duplica.
        """
        for i in range(0, len(embedded_chunks), batch_size):
            batch = embedded_chunks[i:i + batch_size]
            self.collection.upsert(
                ids=[ec.chunk_id for ec in batch],
                embeddings=[ec.embedding for ec in batch],
                documents=[ec.text for ec in batch],
                metadatas=[_payload(ec, self.embedding_model) for ec in batch],
            )

    def search(self, query_embedding: list[float], k: int = 5, filters: dict | None = None) -> list[dict]:
        """Busca os k chunks mais similares. Retorna dicts com id, text, metadata, score."""
        total = self.collection.count()
        if total == 0:
            return []
        if filters and len(filters) > 1:
            filters = {"$and": [{key: value} for key, value in filters.items()]}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, total),
            where=filters or None,
            include=["documents", "metadatas", "distances"],
        )
        # Chroma retorna distância cosine (0 = idêntico); convertemos para similaridade
        return [
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]

    def get_all(self, page_size: int = 5000) -> list[dict]:
        """Todos os chunks (sem vetores) — usado para construir o índice BM25."""
        items, offset = [], 0
        while True:
            page = self.collection.get(include=["documents", "metadatas"], limit=page_size, offset=offset)
            for id_, text, meta in zip(page["ids"], page["documents"], page["metadatas"]):
                items.append({"id": id_, "text": text, "metadata": meta})
            if len(page["ids"]) < page_size:
                return items
            offset += page_size

    def delete_document(self, doc_id: str) -> None:
        """Remove todos os chunks de um documento (para re-indexação)."""
        self.collection.delete(where={"doc_id": doc_id})

    def indexed_documents(self) -> dict[str, dict]:
        """source → {doc_id, content_hash, embedding_model} de tudo que já está indexado."""
        docs = {}
        for item in self.get_all():
            meta = item["metadata"]
            docs.setdefault(meta.get("source"), {
                "doc_id": meta.get("doc_id"),
                "content_hash": meta.get("content_hash"),
                "embedding_model": meta.get("embedding_model"),
            })
        return docs


class QdrantVectorStore:
    """
    Vector store usando Qdrant — recomendado para produção.

    Características:
      - Escrito em Rust, excelente performance
      - Payloads JSON ricos com filtros aplicados durante a busca no HNSW
      - Multi-tenancy via filtros (ex: tenant_id)

    Requer Qdrant rodando:
      docker run -p 6333:6333 qdrant/qdrant
    """

    def __init__(
        self,
        collection_name: str = "rag_kb",
        url: str = "http://localhost:6333",
        vector_size: int = 1024,
        embedding_model: str = "",
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 128,
        hnsw_ef_search: int = 64,
        **_,
    ):
        self.collection_name = collection_name
        self.url = url
        self.vector_size = vector_size
        self.embedding_model = embedding_model
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError:
                raise ImportError("pip install qdrant-client")
            self._client = QdrantClient(url=self.url)
            self._ensure_collection()
        return self._client

    def _ensure_collection(self):
        from qdrant_client.models import Distance, HnswConfigDiff, PayloadSchemaType, VectorParams

        if self._client.collection_exists(self.collection_name):
            return
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=self.hnsw_m, ef_construct=self.hnsw_ef_construction),
        )
        # Índices de payload deixam os filtros eficientes
        for field in ("doc_id", "source", "file_type"):
            self._client.create_payload_index(self.collection_name, field, PayloadSchemaType.KEYWORD)

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        # Qdrant exige int ou UUID; uuid5 é determinístico (hash() do Python não é entre execuções)
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def _filter(self, filters: dict | None):
        if not filters:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        return Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()])

    @staticmethod
    def _to_result(point, score: float | None = None) -> dict:
        payload = dict(point.payload or {})
        text = payload.pop("text", "")
        return {"id": payload.get("chunk_id", str(point.id)), "text": text, "metadata": payload, "score": score}

    def upsert(self, embedded_chunks: list[EmbeddedChunk], batch_size: int = 256) -> None:
        from qdrant_client.models import PointStruct

        for i in range(0, len(embedded_chunks), batch_size):
            points = [
                PointStruct(
                    id=self._point_id(ec.chunk_id),
                    vector=ec.embedding,
                    payload={"text": ec.text, **_payload(ec, self.embedding_model)},
                )
                for ec in embedded_chunks[i:i + batch_size]
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_embedding: list[float], k: int = 5, filters: dict | None = None) -> list[dict]:
        from qdrant_client.models import SearchParams

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=self._filter(filters),
            limit=k,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=self.hnsw_ef_search),
        )
        return [self._to_result(p, p.score) for p in response.points]

    def get_all(self, page_size: int = 1000) -> list[dict]:
        items, offset = [], None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name, limit=page_size, offset=offset,
                with_payload=True, with_vectors=False,
            )
            items.extend(self._to_result(p) for p in points)
            if offset is None:
                return items

    def delete_document(self, doc_id: str) -> None:
        from qdrant_client.models import FilterSelector
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=self._filter({"doc_id": doc_id})),
        )

    def indexed_documents(self) -> dict[str, dict]:
        docs = {}
        for item in self.get_all():
            meta = item["metadata"]
            docs.setdefault(meta.get("source"), {
                "doc_id": meta.get("doc_id"),
                "content_hash": meta.get("content_hash"),
                "embedding_model": meta.get("embedding_model"),
            })
        return docs


def get_vector_store(backend: str = "chroma", **kwargs):
    """Factory para escolher o vector store."""
    stores = {
        "chroma": ChromaVectorStore,
        "qdrant": QdrantVectorStore,
    }
    if backend not in stores:
        raise ValueError(f"Backend '{backend}' desconhecido. Opções: {list(stores)}")
    return stores[backend](**kwargs)
