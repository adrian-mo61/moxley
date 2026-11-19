"""Small, dependency-free JSONL text reader shared by tokenizer and data code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def iter_jsonl_texts(
    path: str | Path,
    max_docs: int = 0,
    min_chars: int = 0,
    text_key: str = "text",
    max_chars: int = 0,
    max_doc_chars: int = 0,
) -> Iterator[str]:
    """Yield text from ``{"text": ...}`` or ``{"conversations": ...}`` JSONL.

    ``max_chars`` caps the total number of characters consumed, which keeps BPE
    training memory bounded on an 8 GB corpus. ``max_doc_chars`` truncates
    unusually long records; both limits are disabled by default.
    """
    path = Path(path)
    used = 0
    chars_used = 0
    line_number = 0
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_docs and used >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if text_key in record and record[text_key] is not None:
                text = str(record[text_key])
            elif "conversations" in record:
                parts = [
                    str(message.get("content", ""))
                    for message in record.get("conversations", [])
                    if message.get("content")
                ]
                text = "\n".join(parts)
            else:
                continue

            # A NUL byte is legal UTF-8 but adds no useful training signal and
            # can confuse downstream tools that treat files as C strings.
            text = text.replace("\x00", "")
            if max_doc_chars and len(text) > max_doc_chars:
                text = text[:max_doc_chars]
            text = text.strip()
            if len(text) < min_chars:
                continue
            if max_chars:
                remaining = max_chars - chars_used
                if remaining <= 0:
                    break
                text = text[:remaining]
            used += 1
            chars_used += len(text)
            yield text

    if used == 0:
        raise ValueError(
            f"no usable text found in {path} (line {line_number}); "
            "expected JSONL objects with a 'text' field"
        )


__all__ = ["iter_jsonl_texts"]
