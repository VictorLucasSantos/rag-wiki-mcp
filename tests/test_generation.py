from types import SimpleNamespace

from src.generation.generator import NOT_FOUND, RAGGenerator, cited_sources
from src.llm import AnthropicLLM, LLMResponse, OllamaLLM, parse_json
from src.generation.prompt_builder import build_messages, fit_to_budget, reorder_for_lost_in_middle


def make_docs(n, size=10):
    return [{"id": str(i), "text": "x" * size, "metadata": {"file_name": f"doc{i}.md"}} for i in range(1, n + 1)]


def test_lost_in_the_middle_puts_best_docs_at_edges():
    order = [d["id"] for d in reorder_for_lost_in_middle(make_docs(5))]
    assert order == ["1", "3", "5", "4", "2"]


def test_budget_keeps_most_relevant_and_at_least_one():
    assert [d["id"] for d in fit_to_budget(make_docs(5, size=100), 250)] == ["1", "2"]
    assert len(fit_to_budget(make_docs(2, size=1000), 10)) == 1


def test_build_messages_numbers_sources_and_wraps_question():
    system, content, sources = build_messages("Qual a regra?", make_docs(3), reorder=False)
    assert "Não encontrei" in system
    assert '<documento id="1" fonte="doc1.md">' in content
    assert "<pergunta>\nQual a regra?\n</pergunta>" in content
    assert [s["label"] for s in sources] == ["doc1.md", "doc2.md", "doc3.md"]


def test_cited_sources():
    sources = [{"n": 1}, {"n": 2}, {"n": 3}]
    assert cited_sources("A [1] e B [3][1].", sources) == [{"n": 1}, {"n": 3}]


class FakeLLM:
    def __init__(self, text="São 30 dias [2]."):
        self.text = text
        self.calls = []

    def complete(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return LLMResponse(text=self.text, model="fake")


def test_generator_maps_citations():
    llm = FakeLLM()
    result = RAGGenerator(llm).generate("Quantos dias?", make_docs(3))
    prompt, kwargs = llm.calls[0]
    assert "<pergunta>" in prompt and "Não encontrei" in kwargs["system"]
    assert result["answer"] == "São 30 dias [2]."
    assert [s["n"] for s in result["cited"]] == [2]


def test_generator_skips_llm_without_docs():
    llm = FakeLLM()
    assert RAGGenerator(llm).generate("Qualquer coisa?", [])["answer"] == NOT_FOUND
    assert llm.calls == []


def test_anthropic_request_shape(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    sent = {}

    def create(**params):
        sent.update(params)
        return SimpleNamespace(
            stop_reason="end_turn", model=params["model"],
            usage=SimpleNamespace(to_dict=lambda: {}),
            content=[SimpleNamespace(type="text", text="ok")],
        )

    llm = AnthropicLLM(model="claude-opus-5-5", effort="low")
    llm.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert llm.complete("pergunta", system="sys", prefix="DOC").text == "ok"
    assert sent["output_config"] == {"effort": "low"}
    assert sent["fallbacks"] == "default"
    assert "temperature" not in sent  # rejeitado pelos modelos atuais
    first_block = sent["messages"][0]["content"][0]
    assert first_block["text"] == "DOC" and first_block["cache_control"] == {"type": "ephemeral"}


def test_ollama_request_shape_and_think_tags():
    sent = {}

    def chat(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(
            message=SimpleNamespace(content="<think>hmm</think> Resposta [1]"),
            prompt_eval_count=100, eval_count=5, done_reason="stop",
        )

    llm = OllamaLLM(model="qwen3.5:4b", num_ctx=8192)
    llm.client = SimpleNamespace(chat=chat)
    response = llm.complete("pergunta", system="sys", schema={"type": "object"})
    assert response.text == "Resposta [1]"
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert sent["options"]["num_ctx"] == 8192
    assert sent["format"] == {"type": "object"}


def test_parse_json_strips_code_fences():
    assert parse_json(LLMResponse(text='```json\n{"a": 1}\n```')) == {"a": 1}
