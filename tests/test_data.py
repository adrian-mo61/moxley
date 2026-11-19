from pathlib import Path

import numpy as np
import torch

from moxley.data import TokenDataset, load_metadata, prepare_bin
from moxley.tokenizer import BBPETokenizer

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample.jsonl"


def test_prepare_bin_and_dataset(tmp_path: Path):
    tokenizer = BBPETokenizer.train(
        SAMPLE, vocab_size=512, max_docs=30, show_progress=False
    )
    tokenizer.save(tmp_path / "tokenizer")
    metadata = prepare_bin(
        SAMPLE,
        tokenizer,
        tmp_path / "bin",
        val_ratio=0.2,
        max_docs=20,
        batch_docs=4,
        quiet=True,
    )
    assert metadata["format"] == "moxley-bin-v1"
    assert metadata["stats"]["documents"] == 20
    assert metadata["stats"]["train_tokens"] > 0
    assert metadata["stats"]["val_tokens"] > 0
    assert (tmp_path / "bin" / "train.bin").stat().st_size == (
        metadata["stats"]["train_tokens"] * 2
    )
    assert (tmp_path / "bin" / "val.bin").stat().st_size == (
        metadata["stats"]["val_tokens"] * 2
    )

    loaded = load_metadata(tmp_path / "bin")
    assert loaded["vocab_size"] == 512
    assert loaded["bos_token_id"] == 1
    assert loaded["eos_token_id"] == 2

    dataset = TokenDataset(tmp_path / "bin" / "train.bin", seq_len=8)
    assert len(dataset) > 0
    x, y = dataset[0]
    assert x.shape == (8,)
    assert y.shape == (8,)
    assert x.dtype == torch.long
    assert torch.equal(x[1:], y[:-1])

    raw = np.memmap(tmp_path / "bin" / "train.bin", dtype=np.uint16, mode="r")
    assert int(raw[0]) == tokenizer.bos_token_id
