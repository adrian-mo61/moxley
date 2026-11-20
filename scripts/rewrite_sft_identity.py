#!/usr/bin/env python
"""Rewrite MiniMind identity statements and build SFT shards in one pass.

The original corpus is not copied to a second JSONL file. Each conversation is
transformed in memory just before tokenization, so the full 14 GB source is read
once and only the final binary shards are written.

Examples
--------
# Quick preview on 1000 conversations
python scripts/rewrite_sft_identity.py --max-samples 1000 --output-dir out/sft_moxley_1k

# Full conversion
python scripts/rewrite_sft_identity.py --output-dir data/sft_bin_moxley
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moxley.identity import rewrite_conversations  # noqa: E402
from moxley.sft import prepare_sft  # noqa: E402
from moxley.tokenizer import BBPETokenizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite model identity and build Moxley SFT shards"
    )
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data" / "sft_t2t.jsonl"
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=ROOT / "data" / "tokenizer"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data" / "sft_bin_moxley"
    )
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--val-ratio", type=float, default=0.005)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-docs", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"input file does not exist: {args.input}")
    if not args.tokenizer.exists():
        parser.error(f"tokenizer does not exist: {args.tokenizer}")

    targets = ["tokens.bin", "mask.bin", "offsets.npy", "meta.json"]
    if any((args.output_dir / name).exists() for name in targets) and not args.overwrite:
        parser.error(f"{args.output_dir} already contains shards; pass --overwrite")

    stats = {
        "records": 0,
        "changed_records": 0,
        "changed_fields": 0,
        "minimind_remaining": 0,
    }

    def transform(conversations: list[dict]) -> list[dict]:
        stats["records"] += 1
        rewritten = rewrite_conversations(conversations, stats)
        if rewritten != conversations:
            stats["changed_records"] += 1
        return rewritten

    tokenizer = BBPETokenizer.from_pretrained(args.tokenizer)
    metadata = prepare_sft(
        input_path=args.input,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        max_len=args.max_len,
        val_ratio=args.val_ratio,
        max_samples=args.max_samples,
        batch_docs=args.batch_docs,
        transform=transform,
    )
    metadata["identity_rewrite"] = {
        "developer": "莫小荒",
        "model": "moxley",
        "original_terms": ["MiniMind", "jingyaogong"],
        "stats": stats,
    }
    with open(args.output_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(
        f"identity rewrite: records={stats['records']:,}, "
        f"changed_records={stats['changed_records']:,}, "
        f"changed_fields={stats['changed_fields']:,}, "
        f"residual_terms={stats['minimind_remaining']:,}",
        flush=True,
    )
    print(json.dumps(metadata["identity_rewrite"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
