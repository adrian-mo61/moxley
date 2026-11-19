from pathlib import Path

import numpy as np
import torch

from moxley.sft import (
    SFTBinDataset,
    build_chat_prompt,
    collate_sft,
    encode_sft_example,
    iter_sft_conversations,
    prepare_sft,
    render_conversation,
)
from moxley.tokenizer import BBPETokenizer

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_sft.jsonl"
TOKENIZER = ROOT / "data" / "tokenizer"


def get_tokenizer() -> BBPETokenizer:
    return BBPETokenizer.from_pretrained(TOKENIZER)


def test_chat_format_and_mask():
    tokenizer = get_tokenizer()
    conversations = list(iter_sft_conversations(SAMPLE))
    text, spans = render_conversation(conversations[0])
    assert text.startswith("<|im_start|>user\n")
    assert "<|im_start|>assistant\n" in text
    assert spans

    encoded = encode_sft_example(tokenizer, conversations[0], max_len=512)
    assert encoded is not None
    ids, mask, chars = encoded
    assert chars > 0
    assert int(mask.sum()) > 0
    supervised = [token for token, keep in zip(ids, mask) if keep]
    decoded = tokenizer.decode(supervised)
    assert "Moxley" in decoded
    assert "请用一句话" not in decoded


def test_tool_call_is_supervised():
    tokenizer = get_tokenizer()
    conversations = list(iter_sft_conversations(SAMPLE))
    ids, mask, _ = encode_sft_example(tokenizer, conversations[4], max_len=1024)
    decoded = tokenizer.decode([token for token, keep in zip(ids, mask) if keep])
    assert "get_weather" in decoded
    assert "Hangzhou" in decoded
    assert "22" in decoded


def test_prepare_sft_and_dataset(tmp_path: Path):
    tokenizer = get_tokenizer()
    output = tmp_path / "sft_bin"
    metadata = prepare_sft(
        SAMPLE,
        tokenizer,
        output,
        max_len=512,
        val_ratio=0.2,
        quiet=True,
    )
    assert metadata["format"] == "moxley-sft-bin-v1"
    assert metadata["stats"]["samples"] == 6
    assert metadata["stats"]["train_samples"] == 4
    assert metadata["stats"]["val_samples"] == 2
    assert metadata["stats"]["tokens"] > 0
    assert (output / "tokens.bin").stat().st_size == metadata["stats"]["tokens"] * 2
    assert (output / "mask.bin").stat().st_size == metadata["stats"]["tokens"]

    train = SFTBinDataset(
        output / "tokens.bin",
        output / "mask.bin",
        output / "offsets.npy",
        output / "train_indices.npy",
    )
    assert len(train) == 4

    batch = [train[i] for i in range(len(train))]
    input_ids, labels = collate_sft(batch)
    assert input_ids.shape == labels.shape
    assert (labels == -100).any()
    assert (labels != -100).any()
    assert torch.equal(labels[labels != -100], input_ids[labels != -100])

    offsets = np.load(output / "offsets.npy")
    assert len(offsets) == metadata["stats"]["samples"] + 1
    assert int(offsets[-1]) == metadata["stats"]["tokens"]


def test_chat_prompt():
    prompt = build_chat_prompt("你好", system="你是助手。")
    assert prompt.startswith("<|im_start|>system\n你是助手。")
    assert prompt.endswith("<|im_start|>assistant\n")
