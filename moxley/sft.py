"""Chat formatting and binary dataset utilities for supervised fine-tuning.

The SFT stage uses ChatML:

    <|im_start|>system
    ...
    <|im_end|>
    <|im_start|>user
    ...
    <|im_end|>
    <|im_start|>assistant
    ...
    <|im_end|>

Only assistant tokens (including the closing ``<|im_end|>``) contribute to the
loss. The tokenizer's character offsets are used to build a per-token mask, so
BPE merges never shift the boundary between prompt and answer.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from .tokenizer import BBPETokenizer


def iter_sft_conversations(
    path: str | Path,
    max_samples: int = 0,
) -> Iterator[list[dict]]:
    """Yield conversation lists from MiniMind-style SFT JSONL."""
    path = Path(path)
    used = 0
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if max_samples and used >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            conversations = record.get("conversations") or record.get("messages")
            if not conversations or not isinstance(conversations, list):
                continue
            used += 1
            yield conversations
    if used == 0:
        raise ValueError(
            f"no usable conversations found in {path}; expected a "
            "'conversations' or 'messages' array"
        )


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _parse_json(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _format_tools(tools: object) -> str:
    tools = _parse_json(tools)
    if not tools:
        return ""
    if not isinstance(tools, list):
        tools = [tools]

    lines = [
        "",
        "# Tools",
        "",
        "You may call one or more functions to assist with the user query.",
        "You are provided with function signatures within <tools></tools> XML tags:",
        "<tools>",
    ]
    for tool in tools:
        lines.append(json.dumps(tool, ensure_ascii=False, separators=(",", ":")))
    lines.extend(
        [
            "</tools>",
            "",
            "For each function call, return a json object with function name and "
            "arguments within <tool_call></tool_call> XML tags:",
            '<tool_call>{"name": <function-name>, "arguments": <args-json-object>}</tool_call>',
        ]
    )
    return "\n".join(lines)


def _format_tool_call(tool_call: object) -> str:
    tool_call = _parse_json(tool_call)
    if not isinstance(tool_call, dict):
        return f"<tool_call>\n{_as_text(tool_call)}\n</tool_call>"
    function = tool_call.get("function", tool_call)
    name = function.get("name", "")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        args_text = arguments
    else:
        args_text = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    return f'<tool_call>\n{{"name": "{name}", "arguments": {args_text}}}\n</tool_call>'


def render_conversation(
    conversations: Sequence[dict],
    add_generation_prompt: bool = False,
) -> tuple[str, list[tuple[int, int]]]:
    """Render ChatML and return ``(text, trainable_character_spans)``."""
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0

    def add(text: str, trainable: bool = False) -> None:
        nonlocal cursor
        if not text:
            return
        start = cursor
        parts.append(text)
        cursor += len(text)
        if trainable:
            spans.append((start, cursor))

    messages = list(conversations)
    if messages and messages[0].get("role") == "system":
        system = messages.pop(0)
        content = _as_text(system.get("content"))
        tools = system.get("tools")
        if tools:
            content += _format_tools(tools)
        add(f"<|im_start|>system\n{content}<|im_end|>\n")

    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role", "user")
        content = _as_text(message.get("content"))

        if role == "assistant":
            add("<|im_start|>assistant\n")
            reasoning = _as_text(message.get("reasoning_content")).strip()
            if reasoning and not content.lstrip().startswith("<think>"):
                add(f"<think>\n{reasoning}\n</think>\n\n", trainable=True)
            add(content, trainable=True)

            tool_calls = _parse_json(message.get("tool_calls"))
            if tool_calls:
                if not isinstance(tool_calls, list):
                    tool_calls = [tool_calls]
                for call_index, tool_call in enumerate(tool_calls):
                    prefix = "\n" if (content or call_index) else ""
                    add(prefix + _format_tool_call(tool_call), trainable=True)

            add("<|im_end|>\n", trainable=True)
            index += 1

        elif role == "tool":
            # Merge consecutive tool responses into one user turn, matching the
            # MiniMind/Qwen tool template.
            add("<|im_start|>user")
            while index < len(messages) and messages[index].get("role") == "tool":
                tool_content = _as_text(messages[index].get("content"))
                add(f"\n<tool_response>\n{tool_content}\n</tool_response>")
                index += 1
            add("<|im_end|>\n")

        elif role in {"user", "system"}:
            add(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            index += 1

        else:
            # Unknown role: skip it instead of silently adding an invalid turn.
            index += 1

    if add_generation_prompt:
        add("<|im_start|>assistant\n")

    return "".join(parts), spans


def build_chat_prompt(
    prompt: str,
    system: str | None = None,
) -> str:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    text, _ = render_conversation(messages, add_generation_prompt=True)
    return text


def encode_sft_example(
    tokenizer: "BBPETokenizer",
    conversations: Sequence[dict],
    max_len: int = 1024,
) -> tuple[list[int], np.ndarray, int] | None:
    """Tokenize one conversation and return ids, 0/1 mask, char count.

    Returns ``None`` when the example has no assistant token in the first
    ``max_len`` tokens.
    """
    text, spans = render_conversation(conversations)
    encoding = tokenizer.tokenizer.encode(text, add_special_tokens=False)
    ids = list(encoding.ids)
    offsets = list(encoding.offsets)
    mask = np.zeros(len(ids), dtype=np.uint8)

    for token_index, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        for span_start, span_end in spans:
            if start < span_end and end > span_start:
                mask[token_index] = 1
                break

    ids = ids[:max_len]
    mask = mask[:max_len]
    if not ids or int(mask.sum()) == 0:
        return None
    return ids, mask, len(text)


class SFTBinDataset(Dataset):
    """Memory-mapped variable-length SFT samples.

    ``tokens.bin`` is uint16, ``mask.bin`` is uint8, and ``offsets.npy`` has
    ``num_samples + 1`` int64 entries.
    """

    def __init__(
        self,
        tokens_path: str | Path,
        mask_path: str | Path,
        offsets_path: str | Path,
        indices_path: str | Path | None = None,
    ) -> None:
        self.tokens_path = str(tokens_path)
        self.mask_path = str(mask_path)
        self.offsets_path = str(offsets_path)
        self._tokens = None
        self._mask = None
        self.offsets = np.load(offsets_path, mmap_mode="r")
        if len(self.offsets) < 2:
            raise ValueError("offsets.npy contains no samples")
        if indices_path is not None:
            self.indices = np.load(indices_path).astype(np.int64)
        else:
            self.indices = np.arange(len(self.offsets) - 1, dtype=np.int64)
        if len(self.indices) == 0:
            raise ValueError("SFT split is empty")

    @property
    def tokens(self) -> np.memmap:
        if self._tokens is None:
            self._tokens = np.memmap(self.tokens_path, dtype="<u2", mode="r")
        return self._tokens

    @property
    def mask(self) -> np.memmap:
        if self._mask is None:
            self._mask = np.memmap(self.mask_path, dtype=np.uint8, mode="r")
        return self._mask

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_tokens"] = None
        state["_mask"] = None
        return state

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_index = int(self.indices[index])
        start = int(self.offsets[sample_index])
        end = int(self.offsets[sample_index + 1])
        ids = torch.from_numpy(
            np.asarray(self.tokens[start:end], dtype=np.int64)
        )
        mask = torch.from_numpy(
            np.asarray(self.mask[start:end], dtype=np.bool_)
        )
        labels = torch.where(mask, ids, torch.full_like(ids, -100))
        return ids, labels


def collate_sft(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a batch; padding labels stay -100 so they never contribute."""
    max_len = max(int(ids.numel()) for ids, _ in batch)
    input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for row, (ids, row_labels) in enumerate(batch):
        length = int(ids.numel())
        input_ids[row, :length] = ids
        labels[row, :length] = row_labels
    return input_ids, labels


def _write_arrays(
    tokens_handle,
    mask_handle,
    token_buffer: list[np.ndarray],
    mask_buffer: list[np.ndarray],
) -> int:
    if not token_buffer:
        return 0
    tokens = np.concatenate(token_buffer)
    masks = np.concatenate(mask_buffer)
    tokens_handle.write(tokens.tobytes())
    mask_handle.write(masks.tobytes())
    written = int(tokens.size)
    token_buffer.clear()
    mask_buffer.clear()
    return written


def prepare_sft(
    input_path: str | Path,
    tokenizer: "BBPETokenizer",
    output_dir: str | Path,
    max_len: int = 1024,
    val_ratio: float = 0.005,
    max_samples: int = 0,
    max_buffer_tokens: int = 32 * 1024 * 1024,
    quiet: bool = False,
) -> dict:
    """Convert SFT JSONL into a uint16 token stream + uint8 loss mask."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if max_len < 2:
        raise ValueError("max_len must be at least 2")
    if tokenizer.vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("uint16 SFT shards require vocab_size <= 65536")

    tokens_path = output_dir / "tokens.bin"
    mask_path = output_dir / "mask.bin"
    offsets_path = output_dir / "offsets.npy"
    train_indices_path = output_dir / "train_indices.npy"
    val_indices_path = output_dir / "val_indices.npy"

    val_every = max(1, round(1 / val_ratio)) if val_ratio > 0 else 0
    offsets = [0]
    split: list[int] = []
    token_buffer: list[np.ndarray] = []
    mask_buffer: list[np.ndarray] = []
    buffered_tokens = 0
    documents = 0
    skipped = 0
    skipped_long = 0
    characters = 0
    started = time.time()

    with open(tokens_path, "wb") as tokens_handle, open(mask_path, "wb") as mask_handle:
        for conversations in iter_sft_conversations(input_path, max_samples=max_samples):
            documents += 1
            example = encode_sft_example(tokenizer, conversations, max_len=max_len)
            if example is None:
                skipped += 1
                continue
            ids, mask, text_chars = example
            if len(ids) >= max_len:
                skipped_long += 1
            token_buffer.append(np.asarray(ids, dtype="<u2"))
            mask_buffer.append(np.asarray(mask, dtype=np.uint8))
            buffered_tokens += len(ids)
            offsets.append(offsets[-1] + len(ids))
            split.append(1 if val_every and (len(split) % val_every == 0) else 0)
            characters += text_chars

            if buffered_tokens >= max_buffer_tokens:
                _write_arrays(tokens_handle, mask_handle, token_buffer, mask_buffer)
                buffered_tokens = 0
                if not quiet:
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        f"documents={documents:,} samples={len(split):,} "
                        f"tokens={offsets[-1]:,} chars/s={characters / elapsed:,.0f}",
                        flush=True,
                    )

        _write_arrays(tokens_handle, mask_handle, token_buffer, mask_buffer)

    if len(split) == 0:
        raise ValueError(
            "no SFT examples produced; check that conversations contain assistant turns"
        )

    offsets_array = np.asarray(offsets, dtype=np.int64)
    split_array = np.asarray(split, dtype=np.uint8)
    train_indices = np.flatnonzero(split_array == 0).astype(np.int64)
    val_indices = np.flatnonzero(split_array == 1).astype(np.int64)
    if len(train_indices) == 0:
        raise ValueError("SFT training split is empty")
    if val_ratio > 0 and len(val_indices) == 0:
        raise ValueError("SFT validation split is empty; lower val_ratio or add samples")

    np.save(offsets_path, offsets_array)
    np.save(train_indices_path, train_indices)
    np.save(val_indices_path, val_indices)

    lengths = np.diff(offsets_array)
    train_tokens = int(lengths[train_indices].sum()) if len(train_indices) else 0
    val_tokens = int(lengths[val_indices].sum()) if len(val_indices) else 0
    tokenizer_hash = None
    source_tokenizer = getattr(tokenizer, "source_path", None)
    if source_tokenizer:
        tokenizer_file = Path(source_tokenizer)
        if tokenizer_file.is_dir():
            tokenizer_file = tokenizer_file / "tokenizer.json"
        if tokenizer_file.exists():
            import hashlib

            with open(tokenizer_file, "rb") as handle:
                tokenizer_hash = hashlib.sha256(handle.read()).hexdigest()

    metadata = {
        "format": "moxley-sft-bin-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(input_path),
        "source_bytes": input_path.stat().st_size,
        "tokenizer": str(source_tokenizer) if source_tokenizer else None,
        "tokenizer_sha256": tokenizer_hash,
        "vocab_size": tokenizer.vocab_size,
        "max_len": max_len,
        "val_ratio": val_ratio,
        "dtype": "uint16 tokens + uint8 mask",
        "stats": {
            "documents": documents,
            "samples": len(split),
            "skipped": skipped,
            "skipped_long": skipped_long,
            "characters": characters,
            "tokens": int(offsets_array[-1]),
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
        },
        "files": {
            "tokens": "tokens.bin",
            "mask": "mask.bin",
            "offsets": "offsets.npy",
            "train_indices": "train_indices.npy",
            "val_indices": "val_indices.npy",
        },
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    if not quiet:
        elapsed = max(time.time() - started, 1e-6)
        print(
            f"done: {len(split):,} samples, {offsets_array[-1]:,} tokens "
            f"in {elapsed:.1f}s",
            flush=True,
        )
        print(f"train: {len(train_indices):,} samples / {train_tokens:,} tokens", flush=True)
        print(f"val:   {len(val_indices):,} samples / {val_tokens:,} tokens", flush=True)
    return metadata


__all__ = [
    "SFTBinDataset",
    "build_chat_prompt",
    "collate_sft",
    "encode_sft_example",
    "iter_sft_conversations",
    "prepare_sft",
    "render_conversation",
]
