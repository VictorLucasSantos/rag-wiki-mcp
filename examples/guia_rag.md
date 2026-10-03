# Guia rápido de RAG

Notas internas sobre Retrieval-Augmented Generation, usadas como documento de exemplo.

## Conceito

RAG combina um mecanismo de busca com um modelo de linguagem. Antes de responder, o sistema recupera trechos relevantes de uma base de documentos e os entrega ao modelo junto com a pergunta. Assim a resposta fica ancorada em evidências verificáveis, em vez de depender apenas do que o modelo memorizou no treinamento.

## Pipeline

### Indexação

A indexação roda quando documentos entram ou mudam: carregar e fazer o parse dos arquivos, limpar ruídos como cabeçalhos repetidos, dividir em chunks, gerar embeddings e gravar os vetores com metadados em um banco vetorial.

### Consulta

A cada pergunta, o sistema gera o embedding da consulta, busca os vizinhos mais próximos, opcionalmente combina com busca por palavra-chave e reordena com um re-ranker, e por fim monta o prompt para o modelo gerar a resposta com citações.

## Chunking

O tamanho do chunk é um compromisso: pedaços pequenos deixam a busca mais precisa, mas carregam pouco contexto; pedaços grandes carregam contexto, mas diluem o sinal. A estratégia parent-child resolve parte do problema buscando em pedaços pequenos e entregando ao modelo o trecho maior que os contém.

## Busca híbrida

A busca vetorial entende sinônimos, mas erra termos exatos como códigos e nomes próprios. O BM25 acerta termos exatos, mas não entende paráfrases. A fusão das duas listas por Reciprocal Rank Fusion usa apenas as posições no ranking, somando 1/(k + posição) com k igual a 60.

## Avaliação

Separe a avaliação da busca (hit rate, MRR) da avaliação da geração (fidelidade ao contexto, relevância da resposta). Um golden set com perguntas reais e fontes esperadas é o ponto de partida.
