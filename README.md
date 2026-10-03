# RAG da wiki de engenharia

Núcleo de RAG que vai alimentar um **servidor MCP para o Claude**: o time pergunta
ao Claude sobre o contexto de programação da empresa (tabelas do banco, fluxo de
desenvolvimento de cada chamado), e o MCP entrega o conteúdo da wiki interna.

Baseado no guia [RAG: do conceito à produção](https://lemon.dev.br/pt/blog/rag-guia-completo).
Usa embeddings locais (BAAI/bge-m3, bom em PT-BR), Chroma ou Qdrant como banco
vetorial e busca híbrida (vetorial + BM25 com RRF). Para testes pela linha de
comando, gera respostas com citações usando um modelo local via Ollama ou Claude pela API.

## 🗺️ Próximos passos

- [ ] Servidor MCP com `search_wiki` e `get_page` (o Claude gera a resposta; o MCP só busca)
- [ ] Páginas de tabela: `describe_table`, `search_tables` e `related_tables`, com a estrutura gerada do schema do banco
- [ ] Páginas de processo por tipo de chamado (`get_process`)
- [ ] Conector da wiki (API e webhooks) no lugar da pasta `docs/`
- [ ] Chunking que não corta blocos de código, e BM25 que separa camelCase e snake_case

---

## 🚀 Como usar

```bash
# 1. Ambiente (torch CPU primeiro: é bem menor que o pacote padrão)
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Copie .env.example para .env. O padrão é LLM_BACKEND=ollama com qwen3.5:4b
#    (ollama pull qwen3.5:4b). Para usar Claude: LLM_BACKEND=anthropic + ANTHROPIC_API_KEY

# 3. Coloque seus arquivos em docs/ (.md .txt .pdf .docx .html) e indexe
#    (para testar com os exemplos fictícios: copie examples/*.md para docs/)
python scripts/index_documents.py --source docs

# 4. Pergunte
python scripts/query.py -q "Quantos dias de férias eu tenho?"
python scripts/query.py            # modo interativo

# 5. Avalie
python scripts/evaluate.py         # retrieval (grátis)
python scripts/evaluate.py --generate   # + geração e LLM-as-judge (usa a API)

# Testes
pytest
```

Na primeira execução o bge-m3 (~2 GB) é baixado do Hugging Face.

---

## 📁 Estrutura

```
config.py                      # TODAS as decisões de arquitetura (também via .env)
docs/                          # SEUS documentos a indexar (ignorada pelo git)
examples/                      # 2 documentos fictícios usados pelo eval/golden_set.json
eval/golden_set.json           # perguntas + fontes esperadas + respostas de referência
data/                          # índice Chroma + docstore parent-child (gerado)

src/
  ingestion/                   # FASES 1-5
    loader.py                  # parse de md/txt/html/pdf/docx + hash do conteúdo
    cleaner.py                 # cabeçalho/rodapé repetido, hifenização, ruído
    chunker.py                 # auto | recursive | markdown | parent_child
    contextualizer.py          # Contextual Retrieval (opcional, usa Claude Haiku)
    embedder.py                # bge-m3 / e5 / OpenAI, com prefixos query/doc corretos
  retrieval/                   # FASE 6
    vector_store.py            # Chroma (dev) e Qdrant (produção)
    hybrid_search.py           # BM25 + Reciprocal Rank Fusion ponderado
    query_transform.py         # Multi-Query e HyDE
    reranker.py                # cross-encoder bge-reranker-v2-m3
    parent_store.py            # troca filhos pelos pais (parent-child)
    retriever.py               # orquestra tudo acima
  generation/                  # FASE 7
    prompt_builder.py          # grounding, citações [n], lost-in-the-middle, orçamento
    generator.py               # gera a resposta via src/llm.py (streaming)
  evaluation/metrics.py        # hit rate, MRR, recall + faithfulness/relevância (LLM-as-judge)
  llm.py                       # backends de LLM: Ollama (local) e Anthropic (Claude)
  pipeline.py                  # Indexer (incremental) e RAGPipeline

scripts/                       # index_documents.py, query.py, evaluate.py
tests/                         # testes unitários (sem rede, sem API)
```

---

## 🔄 Pipeline e onde cada ideia do guia foi aplicada

```
INDEXAÇÃO (scripts/index_documents.py)
  loader → cleaner → chunker → [contextualizer] → embedder → vector store
                                                              + parent store

CONSULTA (scripts/query.py)
  pergunta → [multi-query / HyDE] → busca densa + BM25 → RRF
           → [cross-encoder] → [parent-child] → prompt → Claude → resposta com [n]
```

| Ideia do guia | Implementação |
|---|---|
| Parsers por formato | `loader.py` — headers de HTML/DOCX viram Markdown para o chunker estrutural |
| Limpeza | `cleaner.py` — rodapés repetidos (só nas bordas da página), hifenização, números de página |
| Chunking estrutural | `MarkdownChunker` — o caminho da seção ("Férias > Fracionamento") vai no texto e nos metadados |
| Parent-child | `--chunker parent_child` — busca em trechos de 400 chars, entrega trechos de 3000 |
| Contextual Retrieval | `--contextual` — 1-2 frases por chunk geradas pelo LLM auxiliar (com Claude, usa prompt caching do documento) |
| Assimetria de embedding | `embedder.py` — prefixo certo por família (bge-m3: nenhum; e5: `query:`/`passage:`) |
| Versionamento do embedding | nome do modelo gravado em cada vetor; a indexação recusa misturar modelos (`--rebuild`) |
| HNSW | `M`, `ef_construction`, `ef_search` configuráveis em `VectorStoreConfig` |
| Filtros por metadados | `filters={"file_type": "pdf"}` funciona na busca densa e no BM25 |
| Busca híbrida + RRF | `hybrid_search.py` — BM25 com tokenização sem acentos que mantém IDs como `TI-1180` |
| Multi-Query / HyDE | `--multi-query 3`, `--hyde` |
| Re-ranking | `--rerank` (ou `USE_RERANKER=true`) — busca 30 candidatos e reordena com cross-encoder |
| Lost in the middle | `prompt_builder.py` — ordem [1, 3, 5, 4, 2] |
| Grounding + citações | system prompt com "use só os documentos", "não encontrei", contradições, citações [n] |
| Orçamento de contexto | `max_context_chars` — descarta os menos relevantes primeiro |
| Prompt injection | documentos tratados como dados, não como instruções |
| Recuperação vazia | sem chunks recuperados → responde "não encontrei" sem chamar o LLM |
| Indexação incremental | hash por arquivo: pula os inalterados, substitui os alterados e remove os apagados |
| Avaliação | `evaluate.py` — métricas de retrieval + faithfulness/relevância/correção (estilo RAGAS) |

---

## ⚙️ Ajustes comuns

Tudo fica em `config.py`, e a maior parte também pode ser definida no `.env`:

- **Local x API**: `LLM_BACKEND=ollama` (padrão) ou `anthropic`. Nos dois casos,
  busca e embeddings rodam localmente; só a geração muda.
- **Ollama**: `OLLAMA_MODEL` aceita qualquer modelo baixado com `ollama pull`.
  Modelos maiores (7-14B) seguem melhor as regras de citação e de "não encontrei".
  `OLLAMA_THINK=true` melhora respostas que exigem juntar vários trechos.
  Se aumentar `max_context_chars`, aumente `OLLAMA_NUM_CTX` junto, porque o Ollama corta o excesso sem avisar.
- **Claude**: `LLM_MODEL` (padrão `claude-opus-5-5`) e `LLM_EFFORT`
  (`low` / `medium` / `high`). Para gastar menos, troque por `claude-sonnet-5-5`.
- **Trocou de modelo de embedding ou de chunker?** Rode a indexação com `--rebuild`.
- **PDFs com tabelas**: troque o pypdf por unstructured/pymupdf em `loader.py::_load_pdf`.
- **Produção**: `VECTOR_BACKEND=qdrant` + `pip install qdrant-client` +
  `docker run -p 6333:6333 qdrant/qdrant`.

Antes de mudar qualquer parâmetro de busca, rode `python scripts/evaluate.py`,
faça a mudança e rode de novo. Os resultados ficam em `eval/results/` para comparar.
