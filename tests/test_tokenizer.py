from pathlib import Path

import pytest

from moxley.tokenizer import BBPETokenizer, SPECIAL_TOKENS

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample.jsonl"


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory) -> BBPETokenizer:
    tokenizer = BBPETokenizer.train(
        SAMPLE, vocab_size=512, max_docs=30, show_progress=False
    )
    tokenizer.save(tmp_path_factory.mktemp("tokenizer"))
    return tokenizer


def test_special_token_layout(tokenizer: BBPETokenizer):
    assert len(SPECIAL_TOKENS) == 36
    assert tokenizer.pad_token_id == 0
    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.vocab_size == 512


def test_roundtrip(tokenizer: BBPETokenizer):
    texts = [
        "人工智能正在改变世界。",
        "Hello, world! 你好，世界。",
        "emoji: 🚀🌏  \t newline:\nend",
        "rare: 𠜎 𠮷  \u00e9\u00e8\u00ea",
    ]
    for text in texts:
        ids = tokenizer.encode(text)
        assert tokenizer.decode(ids) == text


def test_save_and_reload(tokenizer: BBPETokenizer, tmp_path: Path):
    tokenizer.save(tmp_path)
    loaded = BBPETokenizer.from_pretrained(tmp_path)
    text = "保存之后重新加载，编码结果应该完全一致。"
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == text
    assert loaded.eos_token_id == tokenizer.eos_token_id
