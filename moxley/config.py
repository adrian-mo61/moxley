"""Model configuration for Moxley-64M."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class MoxleyConfig:
    """Configuration of a Llama/Qwen-style decoder-only Transformer.

    The default values describe Moxley-64M: 63.9M parameters with tied input and
    output embeddings. The architecture follows MiniMind-3: Pre-RMSNorm, SwiGLU,
    RoPE, grouped-query attention, and QK normalization.
    """

    hidden_size: int = 768
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int | None = None
    intermediate_size: int = 2432
    vocab_size: int = 6400
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    model_type: str = "moxley"

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        expected_head_dim = self.hidden_size // self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = expected_head_dim
        if self.head_dim != expected_head_dim:
            raise ValueError(
                f"head_dim={self.head_dim} does not match "
                f"hidden_size / num_attention_heads={expected_head_dim}"
            )
        if self.vocab_size <= max(self.bos_token_id, self.eos_token_id, self.pad_token_id):
            raise ValueError("special token ids must be smaller than vocab_size")

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "MoxleyConfig":
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        allowed = {field for field in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.json", "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Analytically count parameters, including norms and QK norms."""
        h = self.hidden_size
        kv = self.num_key_value_heads * self.head_dim
        per_layer = (
            2 * h * h  # q_proj, o_proj
            + 2 * h * kv  # k_proj, v_proj
            + 2 * self.head_dim  # q_norm, k_norm
            + 3 * h * self.intermediate_size  # SwiGLU
            + 2 * h  # two RMSNorm weights
        )
        total = (
            self.vocab_size * h  # token embedding
            + self.num_hidden_layers * per_layer
            + h  # final norm
        )
        if not self.tie_word_embeddings:
            total += self.vocab_size * h
        if non_embedding:
            total -= self.vocab_size * h
        return total
