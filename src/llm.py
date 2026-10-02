"""
src/llm.py — Camada única de acesso a LLMs (Claude via API ou modelo local via Ollama).

Todo o resto do projeto (gerador, multi-query, HyDE, contextual retrieval,
juiz da avaliação) chama apenas `llm.complete(...)`, então trocar de provedor
é só mudar LLM_BACKEND no .env.

  AnthropicLLM → Claude pela API (pago, melhor qualidade)
  OllamaLLM    → modelo rodando na sua máquina (grátis, dados não saem do PC)
"""

import json
import re
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    text: str
    model: str = ""
    usage: dict = field(default_factory=dict)
    refused: bool = False
    truncated: bool = False


class AnthropicLLM:
    """
    Claude. `effort` controla profundidade de raciocínio/custo; `fallbacks="default"`
    refaz a chamada em um modelo de fallback se o modelo recusar por política.
    """

    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, model: str = "claude-opus-5-5", effort: str | None = None, fallbacks: bool = True):
        import anthropic

        self.model = model
        self.effort = effort
        self.fallbacks = fallbacks
        self.client = anthropic.Anthropic()

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        prefix: str | None = None,
        max_tokens: int = 4000,
        schema: dict | None = None,
        on_text=None,
    ) -> LLMResponse:
        """
        prompt : mensagem do usuário
        prefix : bloco estável colocado ANTES do prompt e marcado para prompt caching
                 (ex.: o documento inteiro no contextual retrieval)
        schema : JSON schema → a resposta é JSON válido nesse formato
        on_text: callback de streaming
        """
        content = [{"type": "text", "text": prompt}]
        if prefix:
            content.insert(0, {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}})

        params = {"model": self.model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}
        if system:
            params["system"] = system
        output_config = {}
        if self.effort:
            output_config["effort"] = self.effort
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if output_config:
            params["output_config"] = output_config
        if self.fallbacks:
            params["betas"] = [self.FALLBACK_BETA]
            params["fallbacks"] = "default"

        messages_api = self.client.beta.messages if self.fallbacks else self.client.messages
        if on_text:
            with messages_api.stream(**params) as stream:
                for text in stream.text_stream:
                    on_text(text)
                message = stream.get_final_message()
        else:
            message = messages_api.create(**params)

        if message.stop_reason == "refusal":
            return LLMResponse(text="", model=message.model, refused=True)
        return LLMResponse(
            text="".join(b.text for b in message.content if b.type == "text").strip(),
            model=message.model,
            usage=message.usage.to_dict(),
            truncated=message.stop_reason == "max_tokens",
        )


_THINK_TAGS = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaLLM:
    """
    Modelo local via Ollama (https://ollama.com). Requer `ollama serve` rodando.

    ATENÇÃO ao num_ctx: o Ollama usa por padrão uma janela de contexto pequena
    e CORTA silenciosamente o que passar dela — o prompt do RAG (instruções +
    documentos) precisa caber. 8192 tokens cobre o orçamento padrão de contexto.
    """

    def __init__(
        self,
        model: str = "qwen3.5:4b",
        host: str = "http://localhost:11434",
        num_ctx: int = 8192,
        temperature: float = 0.2,
        think: bool = False,
    ):
        try:
            import ollama
        except ImportError:
            raise ImportError("pip install ollama")

        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature  # baixa para QA factual
        self.think = think              # raciocínio antes de responder: melhor, porém mais lento
        self.client = ollama.Client(host=host)

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        prefix: str | None = None,
        max_tokens: int = 4000,
        schema: dict | None = None,
        on_text=None,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": f"{prefix}\n\n{prompt}" if prefix else prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "options": {"num_ctx": self.num_ctx, "temperature": self.temperature, "num_predict": max_tokens},
            "think": self.think,
        }
        if schema:
            kwargs["format"] = schema  # saída estruturada nativa do Ollama

        if on_text:
            parts, last = [], None
            for chunk in self.client.chat(stream=True, **kwargs):
                if chunk.message.content:
                    parts.append(chunk.message.content)
                    on_text(chunk.message.content)
                last = chunk
            text, final = "".join(parts), last
        else:
            final = self.client.chat(**kwargs)
            text = final.message.content or ""

        usage = {
            "input_tokens": getattr(final, "prompt_eval_count", None),
            "output_tokens": getattr(final, "eval_count", None),
        }
        if usage["input_tokens"] and usage["input_tokens"] >= self.num_ctx:
            print(f"\n  ⚠️  O prompt encheu a janela de contexto (num_ctx={self.num_ctx}); "
                  "o Ollama pode ter cortado documentos. Aumente OLLAMA_NUM_CTX.")
        return LLMResponse(
            text=_THINK_TAGS.sub("", text).strip(),
            model=self.model,
            usage=usage,
            truncated=getattr(final, "done_reason", None) == "length",
        )


def parse_json(response: LLMResponse) -> dict:
    """Extrai JSON da resposta (modelos locais às vezes envolvem em ```json)."""
    text = response.text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return json.loads(text)


def get_llm(role: str = "main"):
    """
    role: "main" (gera as respostas), "aux" (multi-query, HyDE, contextual) ou "judge".
    Com o Ollama, os três usam o modelo local configurado (podem ser diferentes).
    """
    import config

    lc = config.llm_config
    if lc.backend == "ollama":
        model = {"main": lc.ollama_model, "aux": lc.ollama_aux_model, "judge": lc.ollama_judge_model}[role]
        return OllamaLLM(
            model=model,
            host=lc.ollama_host,
            num_ctx=lc.ollama_num_ctx,
            temperature=lc.ollama_temperature,
            think=lc.ollama_think if role == "main" else False,
        )
    if lc.backend == "anthropic":
        if role == "main":
            return AnthropicLLM(model=lc.model, effort=lc.effort)
        if role == "judge":
            return AnthropicLLM(model=lc.judge_model, effort="low")
        # Haiku 4.5 não suporta effort nem os fallbacks server-side
        return AnthropicLLM(model=lc.aux_model, fallbacks=False)
    raise ValueError(f"LLM_BACKEND '{lc.backend}' desconhecido. Opções: anthropic, ollama")
