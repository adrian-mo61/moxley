"""A compact decoder-only Transformer used by Moxley-64M.

Architecture (following MiniMind-3 / Qwen3-small design choices):

* Pre-RMSNorm residual blocks
* SwiGLU feed-forward network
* Rotary position embeddings (RoPE)
* Grouped-query attention (GQA)
* RMSNorm on Q and K per head
* Tied input/output embeddings
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import MoxleyConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * x).to(dtype)


def precompute_rope(
    head_dim: int,
    max_position_embeddings: int,
    theta: float = 1_000_000.0,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cos/sin tables shaped [max_position_embeddings, head_dim]."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(max_position_embeddings, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q/k: [batch, seq, heads, head_dim]; cos/sin: [seq, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out.to(q.dtype), k_out.to(k.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, T, KVH, D] -> [B, T, KVH * n_rep, D]."""
    if n_rep == 1:
        return x
    batch, seq_len, num_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(batch, seq_len, num_kv_heads, n_rep, head_dim)
        .reshape(batch, seq_len, num_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, config: MoxleyConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rope(q, k, cos, sin)

        past_len = 0
        if past_key_value is not None:
            past_len = past_key_value[0].shape[1]
            k = torch.cat((past_key_value[0], k), dim=1)
            v = torch.cat((past_key_value[1], v), dim=1)

        if seq_len > 1 and past_len > 0:
            raise ValueError("multi-token forward with a KV cache is not supported")

        # Keep the un-repeated K/V for the cache; repeat only for attention.
        k_cache, v_cache = k, v
        q_attn = q.transpose(1, 2)
        k_attn = repeat_kv(k_cache, self.n_rep).transpose(1, 2)
        v_attn = repeat_kv(v_cache, self.n_rep).transpose(1, 2)

        # A single query may attend to every cached key; a full sequence needs a
        # causal mask. SDPA uses an efficient kernel when available.
        output = F.scaled_dot_product_attention(
            q_attn,
            k_attn,
            v_attn,
            dropout_p=0.0,
            is_causal=seq_len > 1,
        )
        output = output.transpose(1, 2).reshape(batch, seq_len, -1)
        present = (k_cache, v_cache) if use_cache else None
        return self.o_proj(output), present


class FeedForward(nn.Module):
    def __init__(self, config: MoxleyConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: MoxleyConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        x, present = self.self_attn(
            self.input_layernorm(x),
            cos,
            sin,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = residual + x
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, present


@dataclass
class CausalLMOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None


class MoxleyModel(nn.Module):
    def __init__(self, config: MoxleyConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        cos, sin = precompute_rope(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[tuple[torch.Tensor, torch.Tensor]]]]:
        _, seq_len = input_ids.shape
        start_pos = 0
        if past_key_values:
            start_pos = past_key_values[0][0].shape[1]
        end_pos = start_pos + seq_len
        if end_pos > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence position {end_pos} exceeds max_position_embeddings="
                f"{self.config.max_position_embeddings}"
            )

        cos = self.rope_cos[start_pos:end_pos]
        sin = self.rope_sin[start_pos:end_pos]
        hidden = self.embed_tokens(input_ids)
        presents: list[tuple[torch.Tensor, torch.Tensor]] = []

        for index, layer in enumerate(self.layers):
            past = past_key_values[index] if past_key_values is not None else None
            hidden, present = layer(hidden, cos, sin, past_key_value=past, use_cache=use_cache)
            if use_cache and present is not None:
                presents.append(present)

        hidden = self.norm(hidden)
        return hidden, (presents if use_cache else None)


class MoxleyForCausalLM(nn.Module):
    def __init__(self, config: MoxleyConfig) -> None:
        super().__init__()
        self.config = config
        self.model = MoxleyModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            total -= self.model.embed_tokens.weight.numel()
        return total

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        hidden, presents = self.model(
            input_ids, past_key_values=past_key_values, use_cache=use_cache
        )
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(loss=loss, logits=logits, past_key_values=presents)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Simple single-batch sampling loop with optional KV cache."""
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        do_sample = temperature is not None and temperature > 0
        past: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
        generated = input_ids
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            if use_cache and past is not None:
                output = self(generated[:, -1:], past_key_values=past, use_cache=True)
            else:
                output = self(generated, use_cache=use_cache)
            logits = output.logits[:, -1].float()

            if do_sample:
                logits = logits / temperature
                if top_k > 0:
                    k = min(top_k, logits.size(-1))
                    threshold = torch.topk(logits, k, dim=-1).values[:, -1, None]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                if 0 < top_p < 1:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    remove = cumulative > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    remove = torch.zeros_like(logits, dtype=torch.bool).scatter(
                        1, sorted_indices, remove
                    )
                    logits = logits.masked_fill(remove, float("-inf"))
                next_token = torch.multinomial(
                    torch.softmax(logits, dim=-1), num_samples=1
                )
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            if finished.any():
                next_token = torch.where(
                    finished[:, None],
                    torch.full_like(next_token, self.config.pad_token_id),
                    next_token,
                )
            generated = torch.cat((generated, next_token), dim=-1)
            past = output.past_key_values
            finished |= next_token.squeeze(-1).eq(eos_token_id)
            if finished.all():
                break

        return generated

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(path)
        torch.save(self.state_dict(), path / "pytorch_model.bin")

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "MoxleyForCausalLM":
        path = Path(path)
        config = MoxleyConfig.from_pretrained(path)
        model = cls(config)
        state_dict = torch.load(path / "pytorch_model.bin", map_location=map_location)
        model.load_state_dict(state_dict)
        return model


__all__ = [
    "Attention",
    "Block",
    "CausalLMOutput",
    "FeedForward",
    "MoxleyForCausalLM",
    "MoxleyModel",
    "RMSNorm",
    "apply_rope",
    "precompute_rope",
    "repeat_kv",
    "rotate_half",
]
