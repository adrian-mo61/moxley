"""JSONL -> uint16 token shards, plus a memory-mapped PyTorch dataset."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import BBPETokenizer, iter_jsonl_texts


@dataclass
class BinStats:
    documents: int = 0
    characters: int = 0
    train_documents: int = 0
    val_documents: int = 0
    train_tokens: int = 0
    val_tokens: int = 0

    @property
    def tokens(self) -> int:
        return self.train_tokens + self.val_tokens


def _write_sequences(handle, sequences: list[np.ndarray]) -> int:
    if not sequences:
        return 0
    block = np.concatenate(sequences)
    handle.write(block.tobytes())
    written = int(block.size)
    sequences.clear()
    return written


def prepare_bin(
    input_path: str | Path,
    tokenizer: BBPETokenizer,
    output_dir: str | Path,
    val_ratio: float = 0.005,
    max_docs: int = 0,
    min_chars: int = 0,
    text_key: str = "text",
    batch_docs: int = 256,
    max_buffer_tokens: int = 32 * 1024 * 1024,
    quiet: bool = False,
) -> dict:
    """Tokenize a JSONL corpus into contiguous ``train.bin`` / ``val.bin``.

    Each document becomes ``[BOS] + text_tokens + [EOS]``. Every token is stored
    as a little-endian uint16, enough for a 6400-token vocabulary and half the
    size of an int32 stream. Validation documents are selected deterministically
    with ``document_index % round(1 / val_ratio) == 0``.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if batch_docs < 1:
        raise ValueError("batch_docs must be positive")

    val_every = max(1, round(1 / val_ratio)) if val_ratio > 0 else 0
    stats = BinStats()
    started = time.time()
    batch_texts: list[str] = []
    train_sequences: list[np.ndarray] = []
    val_sequences: list[np.ndarray] = []
    buffered_tokens = 0
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    if tokenizer.vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("uint16 shards require vocab_size <= 65536")

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"

    def consume_batch() -> None:
        nonlocal buffered_tokens
        if not batch_texts:
            return
        token_lists = tokenizer.encode_batch(batch_texts)
        batch_start = stats.documents - len(batch_texts)
        for offset, token_ids in enumerate(token_lists):
            tokens = np.asarray([bos, *token_ids, eos], dtype="<u2")
            doc_index = batch_start + offset
            if val_every and doc_index % val_every == 0:
                val_sequences.append(tokens)
                stats.val_documents += 1
            else:
                train_sequences.append(tokens)
                stats.train_documents += 1
            buffered_tokens += int(tokens.size)
        batch_texts.clear()

    with open(train_path, "wb") as train_handle, open(val_path, "wb") as val_handle:
        for text in iter_jsonl_texts(
            input_path, max_docs=max_docs, min_chars=min_chars, text_key=text_key
        ):
            stats.documents += 1
            stats.characters += len(text)
            batch_texts.append(text)
            if len(batch_texts) >= batch_docs:
                consume_batch()
            if buffered_tokens >= max_buffer_tokens:
                stats.train_tokens += _write_sequences(train_handle, train_sequences)
                stats.val_tokens += _write_sequences(val_handle, val_sequences)
                buffered_tokens = 0
                if not quiet:
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        f"documents={stats.documents:,} "
                        f"tokens={stats.tokens:,} "
                        f"chars/s={stats.characters / elapsed:,.0f} "
                        f"tokens/s={stats.tokens / elapsed:,.0f}",
                        flush=True,
                    )

        consume_batch()
        stats.train_tokens += _write_sequences(train_handle, train_sequences)
        stats.val_tokens += _write_sequences(val_handle, val_sequences)

    if stats.train_tokens == 0:
        raise ValueError("the training split is empty; add more documents or lower val_ratio")
    if val_ratio > 0 and stats.val_tokens == 0:
        raise ValueError("the validation split is empty; add more documents or lower val_ratio")

    tokenizer_hash = None
    source_tokenizer = getattr(tokenizer, "source_path", None)
    if source_tokenizer:
        tokenizer_file = Path(source_tokenizer)
        if tokenizer_file.is_dir():
            tokenizer_file = tokenizer_file / "tokenizer.json"
        if tokenizer_file.exists():
            with open(tokenizer_file, "rb") as handle:
                tokenizer_hash = hashlib.sha256(handle.read()).hexdigest()
    metadata = {
        "format": "moxley-bin-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dtype": "uint16",
        "endianness": "little",
        "source": str(input_path),
        "source_bytes": input_path.stat().st_size if input_path.exists() else None,
        "tokenizer": str(source_tokenizer) if source_tokenizer else None,
        "tokenizer_sha256": tokenizer_hash,
        "vocab_size": tokenizer.vocab_size,
        "bos_token_id": bos,
        "eos_token_id": eos,
        "pad_token_id": tokenizer.pad_token_id,
        "val_ratio": val_ratio,
        "stats": {**asdict(stats), "tokens": stats.tokens},
        "files": {
            "train": {"name": "train.bin", "tokens": stats.train_tokens},
            "val": {"name": "val.bin", "tokens": stats.val_tokens},
        },
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    if stats.val_tokens == 0:
        val_path.unlink(missing_ok=True)
    if not quiet:
        elapsed = max(time.time() - started, 1e-6)
        print(
            f"done: {stats.documents:,} docs, {stats.tokens:,} tokens in "
            f"{elapsed:.1f}s ({stats.tokens / elapsed:,.0f} tokens/s)",
            flush=True,
        )
        print(f"train.bin: {stats.train_tokens:,} tokens", flush=True)
        print(f"val.bin:   {stats.val_tokens:,} tokens", flush=True)
    return metadata


def load_metadata(data_dir: str | Path) -> dict:
    with open(Path(data_dir) / "meta.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


class TokenDataset(Dataset):
    """Non-overlapping blocks from a uint16 token stream.

    ``__getitem__(i)`` returns ``x = tokens[i*L : (i+1)*L]`` and
    ``y = tokens[i*L+1 : (i+1)*L+1]``. Use a shuffled DataLoader for training.
    The memmap is opened lazily so the dataset can be pickled by DataLoader
    workers on Windows.
    """

    def __init__(self, bin_path: str | Path, seq_len: int) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        self.bin_path = str(bin_path)
        self.seq_len = seq_len
        self._data = None
        self.num_tokens = int(self.data.size)
        self.num_blocks = max(0, (self.num_tokens - 1) // seq_len)
        if self.num_blocks == 0:
            raise ValueError(
                f"{bin_path} has {self.num_tokens} tokens, too few for seq_len={seq_len}"
            )

    @property
    def data(self) -> np.memmap:
        if self._data is None:
            self._data = np.memmap(self.bin_path, dtype="<u2", mode="r")
        return self._data

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_data"] = None
        return state

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.seq_len
        x = torch.from_numpy(self.data[start : start + self.seq_len].astype(np.int64))
        y = torch.from_numpy(
            self.data[start + 1 : start + self.seq_len + 1].astype(np.int64)
        )
        return x, y


__all__ = ["BinStats", "TokenDataset", "load_metadata", "prepare_bin"]
