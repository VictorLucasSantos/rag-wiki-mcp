"""
config.py — Configurações centralizadas do sistema RAG.

Todas as decisões de arquitetura ficam aqui:
- Qual modelo de embedding usar
- Qual vector database usar
- Parâmetros de chunking
- Parâmetros de retrieval (híbrido, re-ranking, transformações de query)
- Parâmetros de geração
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


# ─────────────────────────────────────────────────────────────
# MODELOS
# ─────────────────────────────────────────────────────────────

@dataclass
class EmbeddingConfig:
    """
    Configuração do modelo de embedding.

    Opções:
      backend="local":
        - "BAAI/bge-m3"                    → multilíngue (PT-BR), 1024 dim, sem prefixo
        - "intfloat/multilingual-e5-large" → multilíngue, 1024 dim, prefixos "query:"/"passage:"
        - "BAAI/bge-large-en-v1.5"         → inglês, 1024 dim, instrução só na query
      backend="openai":
        - "text-embedding-3-large"         → 3072 dim, pago, dados saem da infra

    VERSIONAMENTO: o nome do modelo é gravado nos metadados de cada vetor.
    Trocar de modelo exige re-indexação completa (vetores de modelos diferentes
    são incompatíveis) — o script de indexação recusa misturar modelos.
    """
    backend: str = os.getenv("EMBEDDING_BACKEND", "local")
    model_name: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    device: str = os.getenv("EMBEDDING_DEVICE", "cpu")   # "cuda" se tiver GPU
    batch_size: int = 16


@dataclass
class LLMConfig:
    """
    Configuração do modelo de geração de texto.

    backend:
      "ollama"    → modelo local (grátis, privado; qualidade depende do tamanho do modelo)
      "anthropic" → Claude pela API (precisa de ANTHROPIC_API_KEY)

    effort (só Claude) controla profundidade de raciocínio e custo:
      "low"    → respostas rápidas para perguntas factuais simples
      "medium" → padrão do Opus 5.5, bom equilíbrio para QA com RAG
      "high"   → perguntas que exigem síntese de muitos documentos
    """
    backend: str = os.getenv("LLM_BACKEND", "ollama")

    # ── Ollama ──
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
    ollama_aux_model: str = os.getenv("OLLAMA_AUX_MODEL", os.getenv("OLLAMA_MODEL", "qwen3.5:4b"))
    ollama_judge_model: str = os.getenv("OLLAMA_JUDGE_MODEL", os.getenv("OLLAMA_MODEL", "qwen3.5:4b"))
    # Janela de contexto: o Ollama corta em silêncio o que passar disso
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    ollama_temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
    # Raciocinar antes de responder (modelos com "thinking", ex. qwen3.5): melhor e mais lento
    ollama_think: bool = os.getenv("OLLAMA_THINK", "false").lower() == "true"

    # ── Anthropic ──
    model: str = os.getenv("LLM_MODEL", "claude-opus-5-5")
    # Modelo barato para tarefas auxiliares em volume (multi-query, HyDE,
    # contextual retrieval na indexação).
    aux_model: str = os.getenv("LLM_AUX_MODEL", "claude-haiku-4-5")
    # Juiz da avaliação (LLM-as-judge): precisa ser bom em verificar afirmações
    judge_model: str = os.getenv("LLM_JUDGE_MODEL", "claude-sonnet-5-5")
    max_tokens: int = 16000
    effort: str = os.getenv("LLM_EFFORT", "medium")


# ─────────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────────

@dataclass
class ChunkingConfig:
    """
    Estratégia de chunking.

    TRADE-OFF fundamental:
      chunks pequenos → alta precisão de busca, menos contexto por chunk
      chunks grandes  → mais contexto, menor precisão de busca

    Tamanhos em CARACTERES (≈ 4 caracteres por token em português):
      1500 chars ≈ 375 tokens, 200 chars de overlap ≈ 50 tokens.

    Estratégias:
      "auto"         → markdown para .md, recursive para o resto (recomendado)
      "recursive"    → fixed-size com separadores naturais
      "markdown"     → respeita hierarquia de headers
      "parent_child" → indexa filhos pequenos, entrega pais grandes ao LLM
    """
    strategy: str = os.getenv("CHUNK_STRATEGY", "auto")
    chunk_size: int = 1500
    chunk_overlap: int = 200
    # Parent-child
    child_chunk_size: int = 400
    parent_chunk_size: int = 3000
    # Contextual Retrieval (Anthropic, 2024): gera 1-2 frases situando cada
    # chunk no documento antes de indexar. Melhora muito o recall, mas custa
    # uma chamada de LLM por chunk na indexação.
    contextual_retrieval: bool = os.getenv("CONTEXTUAL_RETRIEVAL", "false").lower() == "true"


# ─────────────────────────────────────────────────────────────
# VECTOR DATABASE
# ─────────────────────────────────────────────────────────────

@dataclass
class VectorStoreConfig:
    """
    Configuração do banco vetorial.

    Opções:
      backend = "chroma"   → local, sem infra, ótimo para dev
      backend = "qdrant"   → self-hosted ou cloud, melhor para produção
                             (docker run -p 6333:6333 qdrant/qdrant)
    """
    backend: str = os.getenv("VECTOR_BACKEND", "chroma")
    collection_name: str = "rag_knowledge_base"
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    chroma_path: str = str(DATA_DIR / "chroma_db")
    # Onde ficam os chunks "pai" da estratégia parent_child
    parent_store_path: str = str(DATA_DIR / "parents.json")

    # Parâmetros HNSW (algoritmo de busca aproximada)
    hnsw_ef_construction: int = 128   # Qualidade do índice (maior = melhor, mais lento)
    hnsw_m: int = 16                  # Conexões por nó (maior = mais memória)
    hnsw_ef_search: int = 64          # Precisão da busca (maior = mais preciso)


# ─────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────

@dataclass
class RetrievalConfig:
    """
    Configuração da estratégia de busca.

    Pipeline: [transformação da query] → busca ampla (dense + BM25)
              → fusão RRF → [re-ranking] → top-k final
    """
    k_initial: int = 30        # Candidatos por estratégia de busca
    k_final: int = 5           # Quantos chunks vão para o prompt
    use_hybrid: bool = True    # Combinar busca semântica + BM25
    use_reranker: bool = os.getenv("USE_RERANKER", "false").lower() == "true"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"   # Cross-encoder multilíngue
    rrf_k: int = 60            # Parâmetro RRF (Reciprocal Rank Fusion)

    # Peso de cada lista na fusão RRF ponderada
    semantic_weight: float = 0.6
    keyword_weight: float = 0.4

    # Transformações de query (cada uma custa uma chamada extra de LLM)
    multi_query: int = 0       # Nº de variações da pergunta (0 = desligado; 3-4 recomendado)
    use_hyde: bool = False     # Hypothetical Document Embeddings

    # Abaixo deste score do reranker o chunk é descartado (só com reranker)
    min_rerank_score: float = 0.0


@dataclass
class GenerationConfig:
    """Montagem do prompt."""
    # Orçamento de contexto em caracteres (~4 chars/token → ~3000 tokens)
    max_context_chars: int = 12000
    # Reordena docs para evitar "lost in the middle"
    reorder_lost_in_middle: bool = True
    # Histórico da conversa: quantas trocas (pergunta + resposta) entram no
    # prompt e quanto espaço elas podem ocupar. Disputa a janela do modelo com
    # os documentos: com Ollama, max_context_chars + max_history_chars precisa
    # caber em OLLAMA_NUM_CTX (~3 chars/token, com folga para as instruções).
    history_turns: int = 3
    max_history_chars: int = 3000


# ─────────────────────────────────────────────────────────────
# INSTÂNCIAS GLOBAIS (uso simples)
# ─────────────────────────────────────────────────────────────

embedding_config = EmbeddingConfig()
llm_config = LLMConfig()
chunking_config = ChunkingConfig()
vector_store_config = VectorStoreConfig()
retrieval_config = RetrievalConfig()
generation_config = GenerationConfig()

# OPENAI_API_KEY só é necessária se EMBEDDING_BACKEND=openai.
# A chave da Anthropic é lida automaticamente pelo SDK (ANTHROPIC_API_KEY).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
