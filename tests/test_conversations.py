import config
from app.conversations import ConversationStore, make_title
from src.pipeline import collection_for


def test_conversation_roundtrip_keeps_order_and_meta(tmp_path):
    store = ConversationStore(tmp_path / "c.db")
    conv = store.create("Quantos dias de férias eu tenho?")
    store.add_message(conv["id"], "user", "Quantos dias de férias eu tenho?")
    store.add_message(conv["id"], "assistant", "30 dias [1].", {"sources": [{"n": 1, "label": "rh.md"}]})

    loaded = store.get(conv["id"])
    assert [m["role"] for m in loaded["messages"]] == ["user", "assistant"]
    assert loaded["messages"][1]["meta"]["sources"][0]["label"] == "rh.md"
    assert loaded["messages"][0]["meta"] == {}
    assert store.list()[0]["messages"] == 2


def test_rename_and_delete_cascade(tmp_path):
    store = ConversationStore(tmp_path / "c.db")
    conv = store.create("a")
    store.add_message(conv["id"], "user", "a")
    assert store.rename(conv["id"], "Férias")
    assert store.get(conv["id"])["title"] == "Férias"
    assert store.delete(conv["id"])
    assert store.get(conv["id"]) is None and store.list() == []
    assert not store.delete(conv["id"])


def test_make_title_collapses_whitespace_and_truncates():
    assert make_title("  uma\n pergunta  ") == "uma pergunta"
    assert len(make_title("x" * 200)) == 60


def test_collection_for_keeps_default_name_and_separates_combinations():
    base = config.vector_store_config.collection_name
    model, strategy = config.embedding_config.model_name, config.chunking_config.strategy
    assert collection_for(model) == collection_for(model, strategy) == base
    names = {collection_for(m, s) for m in (model, "intfloat/multilingual-e5-small") for s in (strategy, "parent_child")}
    assert len(names) == 4
