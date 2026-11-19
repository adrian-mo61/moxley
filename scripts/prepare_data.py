#!/usr/bin/env python
"""Convert pretrain_t2t.jsonl into uint16 train.bin / val.bin shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moxley.data import prepare_bin  # noqa: E402
from moxley.tokenizer import BBPETokenizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Moxley .bin pretraining shards")
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data" / "pretrain_t2t.jsonl"
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=ROOT / "data" / "tokenizer"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "bin")
    parser.add_argument("--val-ratio", type=float, default=0.005)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--min-chars", type=int, default=0)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--batch-docs", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"input file does not exist: {args.input}")
    if not args.tokenizer.exists():
        parser.error(f"tokenizer does not exist: {args.tokenizer}")

    targets = [args.output_dir / "train.bin", args.output_dir / "val.bin"]
    if any(path.exists() for path in targets) and not args.overwrite:
        parser.error(f"{args.output_dir} already contains shards; pass --overwrite")

    tokenizer = BBPETokenizer.from_pretrained(args.tokenizer)
    metadata = prepare_bin(
        input_path=args.input,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        max_docs=args.max_docs,
        min_chars=args.min_chars,
        text_key=args.text_key,
        batch_docs=args.batch_docs,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
