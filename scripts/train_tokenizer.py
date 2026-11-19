#!/usr/bin/env python
"""Train the byte-level BPE tokenizer on pretrain_t2t.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moxley.tokenizer import BBPETokenizer, SPECIAL_TOKENS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Moxley BBPE tokenizer")
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data" / "pretrain_t2t.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "tokenizer"
    )
    parser.add_argument("--vocab-size", type=int, default=6400)
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="use only the first N documents (0 = the whole file)",
    )
    parser.add_argument("--min-chars", type=int, default=0)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=100_000_000,
        help="total character budget for BPE training (0 = unlimited)",
    )
    parser.add_argument(
        "--max-doc-chars",
        type=int,
        default=8192,
        help="truncate each document to this many characters (0 = no limit)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"input file does not exist: {args.input}")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        parser.error(f"{args.output} is not empty; pass --overwrite to replace it")

    started = time.time()
    print(f"training BBPE on {args.input} (vocab={args.vocab_size})", flush=True)
    tokenizer = BBPETokenizer.train(
        args.input,
        vocab_size=args.vocab_size,
        max_docs=args.max_docs,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        max_doc_chars=args.max_doc_chars,
    )
    tokenizer.save(args.output)

    # A small, deterministic sanity check that also gives a useful compression
    # number for the model card.
    samples = [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质。",
        "Large language models learn to predict the next token from context.",
        "Python 与 PyTorch 让训练一个小型语言模型变得非常方便。",
        "emoji: 🚀🌏   whitespace:\t\n  done",
    ]
    report = []
    for text in samples:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        report.append(
            {
                "chars": len(text),
                "tokens": len(ids),
                "chars_per_token": len(text) / max(len(ids), 1),
                "roundtrip": decoded == text,
            }
        )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.input),
        "source_bytes": args.input.stat().st_size,
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": len(SPECIAL_TOKENS),
        "max_docs": args.max_docs,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "max_doc_chars": args.max_doc_chars,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "samples": report,
        "train_seconds": round(time.time() - started, 2),
    }
    with open(args.output / "tokenizer_meta.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
