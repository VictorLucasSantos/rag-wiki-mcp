"""
src/ingestion/embedder.py — Fase 4: Geração de embeddings.

Responsabilidade:
  Transformar cada chunk em um vetor numérico de alta dimensão.
  Textos semanticamente similares ficam próximos no espaço vetorial.

DECISÃO CHAVE: qual modelo usar?
  Modelos open source (BAAI/bge-m3, multilingual-e5):
    + Gratuito após download
    + Dados não saem da sua infra (compliance, LGPD)
    + Funcionam bem em PT-BR
    - Requerem hardware para rodar

  APIs (OpenAI text-embedding-3-large):
    + Simples de operar
    - Custo por token
    - Dados saem da infraestrutura

ASSIMETRIA query × documento:
  Alguns modelos foram treinados com prefixos diferentes para queries e
  documentos. Usar o prefixo errado (ou nenhum) derruba a qualidade.
  Cada família tem a sua convenção — ver EMBEDDING_PREFIXES.

ATENÇÃO: use o MESMO modelo tanto na indexação quanto no retrieval.
Vetores de modelos diferentes são incompatíveis.
"""

from dataclasses import dataclass

from src.ingestion.chunker import Chunk

# (prefixo da query, prefixo do documento) por família de modelo
EMBEDDING_PREFIXES = {
    "bge-m3": ("", ""),  # bge-m3 não usa instrução
    "e5": ("query: ", "passage: "),
    "bge-": ("Represent this sentence for searching relevant passages: ", ""),
}


def prefixes_for(model_name: str) -> tuple[str, str]:
    name = model_name.lower()
    for family, prefixes in EMBEDDING_PREFIXES.items():
        if family in name:
            return prefixes
    return "", ""


@dataclass
class EmbeddedChunk:
    """Chunk com seu vetor de embedding associado."""
    chunk: Chunk
    embedding: list[float]

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def metadata(self) -> dict:
        return self.chunk.metadata

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class LocalEmbedder:
    """
    Embeddings usando modelos open source via sentence-transformers.

    Modelos recomendados:
      "BAAI/bge-m3"                     → multilíngue, ótimo para PT-BR
      "intfloat/multilingual-e5-large"  → multilíngue
      "all-MiniLM-L6-v2"                → pequeno e rápido, qualidade menor (inglês)
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu", batch_size: int = 16):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.query_prefix, self.doc_prefix = prefixes_for(model_name)
        self._model = None

    @property
    def model(self):
        """Lazy loading — só carrega o modelo quando necessário."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("pip install sentence-transformers")
            print(f"  🧠 Carregando embedder {self.model_name} ({self.device})...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_texts(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        prefix = self.query_prefix if is_query else self.doc_prefix
        embeddings = self.model.encode(
            [prefix + t for t in texts],
            normalize_embeddings=True,   # Necessário para cosine similarity
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > self.batch_size,
        )
        return [e.tolist() for e in embeddings]

    def embed_documents(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Gera embeddings para chunks (fase de indexação)."""
        print(f"  ⚡ Gerando embeddings para {len(chunks)} chunks...")
        vectors = self.embed_texts([c.indexed_text for c in chunks])
        return [EmbeddedChunk(chunk=c, embedding=v) for c, v in zip(chunks, vectors)]

    def embed_query(self, query: str) -> list[float]:
        """Gera embedding para uma query (fase de retrieval). Usa o prefixo de query."""
        return self.embed_texts([query], is_query=True)[0]


class OpenAIEmbedder:
    """
    Embeddings via OpenAI API (alternativa ao modelo local).

    Modelo recomendado: text-embedding-3-large (3072 dim, melhor qualidade)
    Modelo econômico:   text-embedding-3-small (1536 dim, ~6x mais barato)
    """

    def __init__(self, model_name: str = "text-embedding-3-large", api_key: str | None = None, batch_size: int = 100, **_):
        self.model_name = model_name
        self.api_key = api_key
        self.batch_size = batch_size
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("pip install openai")
            self._client = OpenAI(api_key=self.api_key or None)
        return self._client

    def embed_texts(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        vectors = []
        # Processa em batches para respeitar limite de tokens da API
        for i in range(0, len(texts), self.batch_size):
            response = self.client.embeddings.create(model=self.model_name, input=texts[i:i + self.batch_size])
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def embed_documents(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        vectors = self.embed_texts([c.indexed_text for c in chunks])
        return [EmbeddedChunk(chunk=c, embedding=v) for c, v in zip(chunks, vectors)]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query], is_query=True)[0]


def get_embedder(backend: str = "local", **kwargs):
    """Factory para escolher o embedder."""
    if backend == "local":
        kwargs.pop("api_key", None)
        return LocalEmbedder(**kwargs)
    elif backend == "openai":
        kwargs.pop("device", None)
        return OpenAIEmbedder(**kwargs)
    else:
        raise ValueError(f"Backend '{backend}' desconhecido.")
