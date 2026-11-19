#!/usr/bin/env python
"""Sample from a trained Moxley checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moxley.model import MoxleyForCausalLM  # noqa: E402
from moxley.tokenizer import BBPETokenizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate text with Moxley")
    parser.add_argument("--model", type=Path, default=ROOT / "out" / "final")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "data" / "tokenizer")
    parser.add_argument("--prompt", default="人工智能的未来是")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    tokenizer = BBPETokenizer.from_pretrained(args.tokenizer)
    model = MoxleyForCausalLM.from_pretrained(args.model, map_location="cpu")
    model.to(device).eval()

    prompt_ids = [tokenizer.bos_token_id, *tokenizer.encode(args.prompt)]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=not args.no_cache,
    )
    new_ids = output[0, len(prompt_ids) :].tolist()
    continuation = tokenizer.decode(new_ids, skip_special_tokens=True)
    print(args.prompt + continuation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
