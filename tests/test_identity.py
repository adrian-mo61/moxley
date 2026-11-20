from pathlib import Path

import torch

from moxley.identity import rewrite_conversations, rewrite_text
from moxley.sft import SFTBinDataset, collate_sft, prepare_sft
from moxley.tokenizer import BBPETokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "data" / "tokenizer"


def test_rewrite_text_variants():
    cases = {
        "我是由jingyaogong开发的高效小参数AI模型。": "我是由莫小荒开发的moxley。",
        "我是MiniMind，由jingyaogong开发的一个高效小参数AI模型。": "我是由莫小荒开发的moxley。",
        "我是一个由Jingyaogong开发的高效AI模型，专注于帮助。": "我是由莫小荒开发的moxley，专注于帮助。",
        "MiniMind基于真实开源模型训练。": "moxley基于真实开源模型训练。",
        "我的开发者是jingyaogong。": "我是由莫小荒开发的moxley。",
        "我是一个AI模型，无法观看电影。": "我是由莫小荒开发的moxley，无法观看电影。",
        "我是由莫小荒开发的moxley。": "我是由莫小荒开发的moxley。",
    }
    for source, expected in cases.items():
        actual, _ = rewrite_text(source)
        assert actual == expected, (source, actual, expected)


def test_rewrite_is_recursive():
    stats = {"changed_fields": 0, "minimind_remaining": 0}
    conversations = [
        {
            "role": "system",
            "content": "你是MiniMind。",
        },
        {
            "role": "assistant",
            "content": "我是由jingyaogong开发的高效小参数AI模型。",
            "reasoning_content": "作为MiniMind，我要回答。",
        },
    ]
    rewritten = rewrite_conversations(conversations, stats)
    assert "MiniMind" not in str(rewritten)
    assert "jingyaogong" not in str(rewritten)
    assert "莫小荒开发的moxley" in rewritten[1]["content"]
    assert stats["changed_fields"] >= 2


def test_prepare_sft_with_identity_transform(tmp_path: Path):
    tokenizer = BBPETokenizer.from_pretrained(TOKENIZER)
    source = tmp_path / "sft.jsonl"
    source.write_text(
        '{"conversations":[{"role":"user","content":"你由谁开发？"},'
        '{"role":"assistant","content":"我是由jingyaogong开发的高效小参数AI模型。"}]}\n'
        '{"conversations":[{"role":"assistant","content":"我是MiniMind，由jingyaogong创建。"}]}\n',
        encoding="utf-8",
    )
    stats = {"changed_fields": 0, "minimind_remaining": 0}
    metadata = prepare_sft(
        source,
        tokenizer,
        tmp_path / "bin",
        max_len=256,
        val_ratio=0.5,
        quiet=True,
        transform=lambda conversations: rewrite_conversations(conversations, stats),
    )
    assert metadata["stats"]["samples"] == 2
    assert stats["changed_fields"] >= 2
    assert stats["minimind_remaining"] == 0

    dataset = SFTBinDataset(
        tmp_path / "bin" / "tokens.bin",
        tmp_path / "bin" / "mask.bin",
        tmp_path / "bin" / "offsets.npy",
        tmp_path / "bin" / "train_indices.npy",
    )
    x, y = collate_sft([dataset[i] for i in range(len(dataset))])
    supervised = x[y != -100]
    decoded = tokenizer.decode(supervised.tolist(), skip_special_tokens=True)
    assert "moxley" in decoded
    assert "jingyaogong" not in decoded
    assert "MiniMind" not in decoded
