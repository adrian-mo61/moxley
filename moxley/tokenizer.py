"""Byte-level BPE tokenizer used by Moxley.

``BBPE`` means **byte-level byte-pair encoding**: every UTF-8 byte is first
mapped to a printable Unicode symbol (the GPT-2 byte-to-unicode table), then BPE
merges are learned over those symbols. Because all 256 byte symbols are in the
initial alphabet, any UTF-8 string can be encoded without an ``<unk>`` token.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# The same 36-slot special-token layout as MiniMind, which keeps the first three
# ids stable across the ecosystem: pad/unk=0, bos=1, eos=2.
SPECIAL_TOKENS: tuple[str, ...] = (
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|audio_pad|>",
    "<tts_pad>",
    "<tts_text_bos>",
    "<tts_text_eod>",
    "<tts_text_bos_single>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    *[f"<|buffer{i}|>" for i in range(1, 10)],
)

PAD_TOKEN = "<|endoftext|>"
BOS_TOKEN = "<|im_start|>"
EOS_TOKEN = "<|im_end|>"


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
            f"no usable text found in {path} (line {line_number if 'line_number' in locals() else 0}); "
            "expected JSONL objects with a 'text' field"
        )


class BBPETokenizer:
    """Thin, explicit wrapper around a Hugging Face ``tokenizers`` BPE model."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.source_path: str | None = None
        self._validate_special_tokens()

    def _validate_special_tokens(self) -> None:
        vocab = self.tokenizer.get_vocab()
        expected = {
            "<|endoftext|>": 0,
            "<|im_start|>": 1,
            "<|im_end|>": 2,
        }
        for token, token_id in expected.items():
            actual = vocab.get(token)
            if actual != token_id:
                raise ValueError(
                    f"tokenizer has {token!r} at id {actual}, expected {token_id}"
                )

    @classmethod
    def train(
        cls,
        data_path: str | Path,
        vocab_size: int = 6400,
        max_docs: int = 0,
        min_chars: int = 0,
        max_chars: int = 0,
        max_doc_chars: int = 0,
        show_progress: bool = True,
    ) -> "BBPETokenizer":
        if vocab_size < len(SPECIAL_TOKENS) + 256:
            raise ValueError(
                f"vocab_size must be at least {len(SPECIAL_TOKENS) + 256} "
                "to hold the 256 byte symbols and special tokens"
            )

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            show_progress=show_progress,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=list(SPECIAL_TOKENS),
        )
        texts: Iterable[str] = iter_jsonl_texts(
            data_path,
            max_docs=max_docs,
            min_chars=min_chars,
            max_chars=max_chars,
            max_doc_chars=max_doc_chars,
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        return cls(tokenizer)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "BBPETokenizer":
        path = Path(path)
        if path.is_dir():
            path = path / "tokenizer.json"
        tokenizer = cls(Tokenizer.from_file(str(path)))
        tokenizer.source_path = str(path.parent)
        return tokenizer

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path / "tokenizer.json"))
        self.source_path = str(path)
        # vocab.json / merges.txt make the BPE table easy to inspect.
        self.tokenizer.model.save(str(path))

        vocab = self.tokenizer.get_vocab()
        added_tokens_decoder = {
            str(vocab[token]): {
                "content": token,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            }
            for token in SPECIAL_TOKENS
        }
        config = {
            "add_bos_token": False,
            "add_eos_token": False,
            "add_prefix_space": False,
            "added_tokens_decoder": added_tokens_decoder,
            "bos_token": BOS_TOKEN,
            "clean_up_tokenization_spaces": False,
            "eos_token": EOS_TOKEN,
            "legacy": True,
            "model_max_length": 32768,
            "pad_token": PAD_TOKEN,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "unk_token": PAD_TOKEN,
        }
        with open(path / "tokenizer_config.json", "w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        special_tokens_map = {
            "bos_token": BOS_TOKEN,
            "eos_token": EOS_TOKEN,
            "pad_token": PAD_TOKEN,
            "unk_token": PAD_TOKEN,
            "additional_special_tokens": [
                token for token in SPECIAL_TOKENS if token not in {BOS_TOKEN, EOS_TOKEN, PAD_TOKEN}
            ],
        }
        with open(path / "special_tokens_map.json", "w", encoding="utf-8") as handle:
            json.dump(special_tokens_map, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.token_to_id(BOS_TOKEN)

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.token_to_id(EOS_TOKEN)

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.token_to_id(PAD_TOKEN)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_batch(
        self, texts: Sequence[str], add_special_tokens: bool = False
    ) -> list[list[int]]:
        encodings = self.tokenizer.encode_batch(
            list(texts), add_special_tokens=add_special_tokens
        )
        return [encoding.ids for encoding in encodings]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def __len__(self) -> int:
        return self.vocab_size


__all__ = [
    "BBPETokenizer",
    "BOS_TOKEN",
    "EOS_TOKEN",
    "PAD_TOKEN",
    "SPECIAL_TOKENS",
    "iter_jsonl_texts",
]
