"""
app/server.py — Interface web do RAG: alimentar a base e fazer perguntas.

  python app/server.py              → http://127.0.0.1:8765

Endpoints:
  GET    /api/status                → backend e opções padrão
  GET    /api/models                → modelos instalados no Ollama (para o seletor)
  GET    /api/index                 → modelos de embedding e chunkers oferecidos, e quais estão em uso
  POST   /api/index                 → troca embedding e/ou chunker (progresso via SSE)
  GET    /api/documents             → documentos indexados (com nº de chunks)
  POST   /api/documents             → upload de arquivos (salva em docs/ e indexa)
  POST   /api/documents/text        → texto colado vira docs/<titulo>.md e é indexado
  DELETE /api/documents?source=...  → remove do índice e apaga o arquivo de docs/
  POST   /api/reindex               → reindexa a pasta docs/ inteira (incremental)
  GET    /api/conversations         → conversas salvas
  GET    /api/conversations/{id}    → uma conversa com as mensagens
  PATCH  /api/conversations/{id}    → renomeia
  DELETE /api/conversations/{id}    → apaga
  POST   /api/ask                   → resposta em streaming (Server-Sent Events), salva na conversa

Embedder e vector store são carregados UMA vez e compartilhados entre o
Indexer e o RAGPipeline (o bge-m3 ocupa ~2 GB). Um lock serializa indexação
e consultas: o Chroma local e o modelo de embedding não são feitos para
acesso concorrente, e o Ollama atende uma geração por vez de qualquer jeito.
"""

import gc
import json
import os
import queue
import re
import sys
import threading
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # source_id() é relativo ao diretório atual: "docs/arquivo.md"
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # logs aparecem na hora

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

import config
from app.conversations import ConversationStore
from src.ingestion.loader import DocumentLoader
from src.pipeline import Indexer, RAGPipeline, build_embedder, build_vector_store, collection_for

DOCS_DIR = ROOT / "docs"
STATIC_DIR = Path(__file__).resolve().parent / "static"
SUPPORTED = DocumentLoader.SUPPORTED_EXTENSIONS

app = FastAPI(title="RAG")
lock = threading.Lock()
_state: dict = {}
conversations = ConversationStore(config.DATA_DIR / "conversations.db")
# Escolhas feitas na interface que precisam sobreviver a um reinício (embedding, chunker)
UI_STATE = config.DATA_DIR / "ui_state.json"


def load_ui_state() -> dict:
    try:
        return json.loads(UI_STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_ui_state(**changes) -> None:
    UI_STATE.parent.mkdir(parents=True, exist_ok=True)
    UI_STATE.write_text(json.dumps({**load_ui_state(), **changes}, ensure_ascii=False), encoding="utf-8")


def embedding_choices() -> dict[str, str]:
    if config.embedding_config.backend != "local":
        return {config.embedding_config.model_name: "modelo do .env"}
    choices = dict(config.EMBEDDING_CHOICES)
    choices.setdefault(config.embedding_config.model_name, "modelo do .env")
    return choices


def current_index() -> tuple[str, str]:
    """(modelo de embedding, chunker) em uso — ou os que serão usados ao carregar."""
    if _state:
        indexer = _state["indexer"]
        return indexer.embedder.model_name, indexer.strategy
    saved = load_ui_state()
    model = saved.get("embedding_model")
    strategy = saved.get("chunker")
    return (
        model if model in embedding_choices() else config.embedding_config.model_name,
        strategy if strategy in config.CHUNKER_CHOICES else config.chunking_config.strategy,
    )


def components() -> tuple[Indexer, RAGPipeline]:
    """Carrega os modelos na primeira requisição que precisar deles."""
    with lock:
        if not _state:
            model, strategy = current_index()
            embedder = build_embedder(model)
            store = build_vector_store(embedder, strategy)
            _state["indexer"] = Indexer(strategy=strategy, embedder=embedder, vector_store=store)
            _state["pipeline"] = RAGPipeline(embedder=embedder, vector_store=store)
    return _state["indexer"], _state["pipeline"]


def safe_filename(name: str) -> str:
    name = Path(name or "").name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    if not name or Path(name).suffix.lower() not in SUPPORTED:
        raise HTTPException(400, f"Formato não suportado: {name or '?'} (aceitos: {', '.join(sorted(SUPPORTED))})")
    return name


def index_paths(paths: list[Path]) -> dict:
    indexer, pipeline = components()
    total = Counter()
    with lock:
        for path in paths:
            total.update(indexer.index(str(path)))
        pipeline.refresh()
    return dict(total)


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_worker(work) -> StreamingResponse:
    """Roda work(emit) numa thread e repassa cada emit(evento, dados) ao navegador como SSE."""
    events: queue.Queue = queue.Queue()

    def run():
        try:
            work(lambda event, data: events.put((event, data)))
        except Exception as e:  # erro chega ao navegador em vez de morrer no log
            events.put(("error", str(e)))
        finally:
            events.put(None)

    threading.Thread(target=run, daemon=True).start()

    def stream():
        while (item := events.get()) is not None:
            yield _sse(*item)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ─────────────────────────────────────────────────────────────
# BASE DE CONHECIMENTO
# ─────────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    lc = config.llm_config
    model, strategy = current_index()
    return {
        "llm_backend": lc.backend,
        "llm_model": lc.ollama_model if lc.backend == "ollama" else lc.model,
        "embedding_model": model,
        "chunker": strategy,
        "vector_backend": config.vector_store_config.backend,
        "defaults": {
            "rerank": config.retrieval_config.use_reranker,
            "hyde": config.retrieval_config.use_hyde,
            "multi_query": config.retrieval_config.multi_query,
        },
        "loaded": bool(_state),
    }


@app.get("/api/documents")
def list_documents():
    indexer, _ = components()
    with lock:
        chunks = indexer.vector_store.get_all()
    docs: dict[str, dict] = {}
    for c in chunks:
        meta = c["metadata"]
        source = meta.get("source") or "?"
        d = docs.setdefault(source, {
            "source": source,
            "file_name": meta.get("file_name") or Path(source).name,
            "file_type": meta.get("file_type"),
            "chunks": 0,
            "chars": 0,
        })
        d["chunks"] += 1
        d["chars"] += len(c["text"] or "")
    for d in docs.values():
        d["in_docs_folder"] = (ROOT / d["source"]).is_file()
    return sorted(docs.values(), key=lambda d: d["file_name"].lower())


@app.post("/api/documents")
def upload_documents(files: list[UploadFile]):
    DOCS_DIR.mkdir(exist_ok=True)
    saved = []
    for f in files:
        dest = DOCS_DIR / safe_filename(f.filename)
        dest.write_bytes(f.file.read())
        saved.append(dest)
    try:
        stats = index_paths(saved)
    except Exception as e:
        raise HTTPException(500, f"Falha ao indexar: {e}")
    return {"saved": [p.name for p in saved], "stats": stats}


class TextDoc(BaseModel):
    title: str
    content: str


@app.post("/api/documents/text")
def add_text(doc: TextDoc):
    if not doc.content.strip():
        raise HTTPException(400, "O texto está vazio.")
    stem = re.sub(r"[^\w\- ]", "", doc.title, flags=re.UNICODE).strip() or "nota"
    DOCS_DIR.mkdir(exist_ok=True)
    dest = DOCS_DIR / f"{stem}.md"
    body = doc.content if doc.content.lstrip().startswith("#") else f"# {doc.title.strip() or stem}\n\n{doc.content}"
    dest.write_text(body, encoding="utf-8")
    return {"saved": [dest.name], "stats": index_paths([dest])}


@app.delete("/api/documents")
def delete_document(source: str, delete_file: bool = True):
    indexer, pipeline = components()
    with lock:
        removed = indexer.remove_source(source)
        pipeline.refresh()
    path = (ROOT / source).resolve()
    file_deleted = False
    # Só apaga arquivos dentro de docs/ — nunca um caminho arbitrário vindo da requisição
    if delete_file and path.is_file() and path.is_relative_to(DOCS_DIR.resolve()):
        path.unlink()
        file_deleted = True
    if not removed and not file_deleted:
        raise HTTPException(404, "Documento não encontrado.")
    return {"removed": removed, "file_deleted": file_deleted}


@app.post("/api/reindex")
def reindex():
    indexer, pipeline = components()
    with lock:
        stats = indexer.index(str(DOCS_DIR.relative_to(ROOT)))
        pipeline.refresh()
    return {"stats": stats}


@app.get("/api/models")
def list_models():
    """Modelos de chat instalados no Ollama. Com outro backend, a lista vem vazia (sem seletor)."""
    lc = config.llm_config
    if lc.backend != "ollama":
        return {"backend": lc.backend, "default": lc.model, "models": []}
    try:
        with urllib.request.urlopen(f"{lc.ollama_host.rstrip('/')}/api/tags", timeout=5) as r:
            tags = json.load(r)["models"]
    except Exception as e:
        raise HTTPException(503, f"Não consegui falar com o Ollama em {lc.ollama_host}: {e}")
    models = [
        {
            "name": m["name"],
            "params": m.get("details", {}).get("parameter_size"),
            "size_gb": round(m.get("size", 0) / 1e9, 1),
        }
        for m in tags
        # Modelos de embedding não conversam: ficam fora do seletor
        if "embed" not in m["name"] and m.get("details", {}).get("family") not in {"bert", "nomic-bert"}
    ]
    return {"backend": "ollama", "default": lc.ollama_model, "models": sorted(models, key=lambda m: m["name"])}


# ─────────────────────────────────────────────────────────────
# ÍNDICE: MODELO DE EMBEDDING + CHUNKER
# ─────────────────────────────────────────────────────────────

def _hf_downloaded(model_name: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(model_name, "modules.json"), str)
    except Exception:
        return False


def _existing_collections() -> set[str] | None:
    """Coleções já criadas no Chroma (None se o backend não for Chroma)."""
    if config.vector_store_config.backend != "chroma":
        return None
    import chromadb
    client = chromadb.PersistentClient(path=config.vector_store_config.chroma_path)
    return {getattr(c, "name", c) for c in client.list_collections()}


@app.get("/api/index")
def index_options():
    """
    Opções dos seletores. `indexed` diz se já existe índice para a combinação
    (opção × o outro valor em uso) — trocar para ela é só uma sincronização.
    """
    model, strategy = current_index()
    existing = _existing_collections()
    has = lambda m, s: None if existing is None else collection_for(m, s) in existing
    local = config.embedding_config.backend == "local"
    return {
        "embedding": {
            "current": model,
            "default": config.embedding_config.model_name,
            # Só embeddings locais têm seletor (trocar o da OpenAI é decisão de custo, fica no .env)
            "models": [
                {"name": name, "description": desc, "downloaded": _hf_downloaded(name), "indexed": has(name, strategy)}
                for name, desc in embedding_choices().items()
            ] if local else [],
        },
        "chunker": {
            "current": strategy,
            "default": config.chunking_config.strategy,
            "options": [
                {"name": name, "description": desc, "indexed": has(model, name)}
                for name, desc in config.CHUNKER_CHOICES.items()
            ],
        },
    }


class IndexChoice(BaseModel):
    embedding_model: str | None = None
    chunker: str | None = None


@app.post("/api/index")
def switch_index(choice: IndexChoice):
    """
    Troca o modelo de embedding e/ou o chunker. Cada combinação tem a sua coleção:
    a pasta docs/ é sincronizada com ela (tudo na primeira vez, só o que mudou nas
    seguintes). O índice anterior só é largado depois que o novo ficou pronto.
    """
    if choice.embedding_model and choice.embedding_model not in embedding_choices():
        raise HTTPException(400, f"Modelo de embedding não disponível: {choice.embedding_model}")
    if choice.chunker and choice.chunker not in config.CHUNKER_CHOICES:
        raise HTTPException(400, f"Chunker desconhecido: {choice.chunker}")
    components()

    def work(emit):
        with lock:
            indexer, pipeline = _state["indexer"], _state["pipeline"]
            model = choice.embedding_model or indexer.embedder.model_name
            strategy = choice.chunker or indexer.strategy
            if (model, strategy) == (indexer.embedder.model_name, indexer.strategy):
                emit("done", {"embedding_model": model, "chunker": strategy, "stats": None})
                return

            if model == indexer.embedder.model_name:
                embedder = indexer.embedder
            else:
                emit("status", f"Carregando {model}"
                     + ("" if _hf_downloaded(model) else " (baixando na primeira vez, pode demorar)") + "…")
                embedder = build_embedder(model)
                embedder.model  # força o carregamento/download aqui, com o aviso na tela

            store = build_vector_store(embedder, strategy)
            new_indexer = Indexer(strategy=strategy, embedder=embedder, vector_store=store)
            emit("status", f"Indexando a pasta docs/ ({model.split('/')[-1]} + {strategy})…"
                 if not store.indexed_documents() else "Sincronizando esse índice com a pasta docs/…")
            stats = new_indexer.index(str(DOCS_DIR.relative_to(ROOT)))

            _state["indexer"] = new_indexer
            pipeline.set_embedding(embedder, store)
            del indexer
            gc.collect()  # libera a memória do modelo anterior, se ele mudou
            save_ui_state(embedding_model=model, chunker=strategy)
        emit("done", {"embedding_model": model, "chunker": strategy, "stats": stats})

    return sse_worker(work)


# ─────────────────────────────────────────────────────────────
# CONVERSAS
# ─────────────────────────────────────────────────────────────

@app.get("/api/conversations")
def list_conversations():
    return conversations.list()


@app.get("/api/conversations/{conv_id}")
def get_conversation(conv_id: str):
    conv = conversations.get(conv_id)
    if not conv:
        raise HTTPException(404, "Conversa não encontrada.")
    return conv


class Rename(BaseModel):
    title: str = Field(min_length=1)


@app.patch("/api/conversations/{conv_id}")
def rename_conversation(conv_id: str, body: Rename):
    if not conversations.rename(conv_id, body.title):
        raise HTTPException(404, "Conversa não encontrada.")
    return {"ok": True}


@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    if not conversations.delete(conv_id):
        raise HTTPException(404, "Conversa não encontrada.")
    return {"ok": True}


# ─────────────────────────────────────────────────────────────
# PERGUNTAS
# ─────────────────────────────────────────────────────────────

class Question(BaseModel):
    question: str
    # Conversa onde salvar; None (ou id inexistente) = cria uma nova
    conversation_id: str | None = None
    # None = usa o padrão do config/.env
    rerank: bool | None = None
    hyde: bool | None = None
    multi_query: int | None = Field(None, ge=0, le=6)  # nº de variações; 0 = desligado
    # Modelo do Ollama para a resposta E para multi-query/HyDE.
    # None = cada um usa o seu do .env (OLLAMA_MODEL / OLLAMA_AUX_MODEL)
    model: str | None = None


@app.post("/api/ask")
def ask(q: Question):
    question = q.question.strip()
    if not question:
        raise HTTPException(400, "Pergunta vazia.")
    _, pipeline = components()

    conv = conversations.get(q.conversation_id) if q.conversation_id else None
    if not conv:
        conv = conversations.create(question)
    conversations.add_message(conv["id"], "user", question)

    def work(emit):
        emit("conversation", {"id": conv["id"], "title": conv["title"]})
        try:
            with lock:
                pipeline.configure(use_reranker=q.rerank, use_hyde=q.hyde, multi_query=q.multi_query)
                if config.llm_config.backend == "ollama":
                    lc = config.llm_config
                    pipeline.generator.llm.model = q.model or lc.ollama_model
                    if pipeline._query_transformer:  # existe se multi-query/HyDE já foram ligados
                        pipeline._query_transformer.llm.model = q.model or lc.ollama_aux_model
                reranker = pipeline.retriever.reranker
                if reranker and reranker._model is None:
                    # O modelo é lazy: carrega aqui para o aviso aparecer durante a espera
                    emit("status", "Carregando o reranker (na primeira vez baixa ~2,3 GB)…")
                    reranker.model
                rc = pipeline.retriever.config
                steps = [f"{rc.multi_query} variações da pergunta"] if rc.multi_query else []
                if rc.use_hyde:
                    steps.append("documento hipotético (HyDE)")
                if steps:
                    emit("status", f"Gerando {' e '.join(steps)}…")
                result = pipeline.ask(question, stream=lambda t: emit("token", t))
                index_used = {"embedding_model": pipeline.embedder.model_name, "chunker": _state["indexer"].strategy}
        except Exception as e:
            conversations.add_message(conv["id"], "assistant", str(e), {"error": True})
            raise

        by_id = {d.get("id"): d for d in result["retrieval"]["docs"]}
        cited = {s["n"] for s in result["cited"]}
        meta = {
            "sources": [
                {**s, "cited": s["n"] in cited, "text": by_id.get(s["id"], {}).get("text", ""),
                 "rerank_score": by_id.get(s["id"], {}).get("rerank_score")}
                for s in result["sources"]
            ],
            "model": result["model"],
            "hyde": result["retrieval"]["hyde"],
            "queries": result["retrieval"]["queries"][1:],
            "options": {"rerank": rc.use_reranker, "hyde": rc.use_hyde, "multi_query": rc.multi_query},
            "index": index_used,
        }
        conversations.add_message(conv["id"], "assistant", result["answer"], meta)
        emit("done", {"answer": result["answer"], **meta})

    return sse_worker(work)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Interface web do RAG")
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 para acessar de outra máquina da rede")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(f"\n🌐 RAG em http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port)
