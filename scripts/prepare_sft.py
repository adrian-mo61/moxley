#!/usr/bin/env python
"""Convert sft_t2t.jsonl into token/mask/offset shards for train_sft.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moxley.sft import prepare_sft  # noqa: E402
from moxley.tokenizer import BBPETokenizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Moxley SFT binary shards")
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data" / "sft_t2t.jsonl"
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=ROOT / "data" / "tokenizer"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data" / "sft_bin"
    )
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--val-ratio", type=float, default=0.005)
    parser.add_argument(
        "--batch-docs",
        type=int,
        default=256,
        help="encode this many conversations per tokenizer batch",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="only read the first N conversations (0 = all)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"input file does not exist: {args.input}")
    if not args.tokenizer.exists():
        parser.error(f"tokenizer does not exist: {args.tokenizer}")

    targets = ["tokens.bin", "mask.bin", "offsets.npy", "meta.json"]
    if any((args.output_dir / name).exists() for name in targets) and not args.overwrite:
        parser.error(f"{args.output_dir} already contains SFT shards; pass --overwrite")

    tokenizer = BBPETokenizer.from_pretrained(args.tokenizer)
    metadata = prepare_sft(
        input_path=args.input,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        max_len=args.max_len,
        val_ratio=args.val_ratio,
        max_samples=args.max_samples,
        batch_docs=args.batch_docs,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
