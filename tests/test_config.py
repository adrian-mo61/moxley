import json
from pathlib import Path

import pytest

from moxley.config import MoxleyConfig


def test_default_config_is_64m():
    config = MoxleyConfig()
    params = config.num_parameters()
    assert 63_000_000 < params < 65_000_000
    assert params == 63_912_192
    assert config.head_dim == config.hidden_size // config.num_attention_heads


def test_config_roundtrip(tmp_path: Path):
    config = MoxleyConfig(vocab_size=1024)
    config.save_pretrained(tmp_path)
    loaded = MoxleyConfig.from_pretrained(tmp_path)
    assert loaded == config
    data = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert data["hidden_size"] == 768


def test_invalid_head_dim():
    with pytest.raises(ValueError, match="head_dim"):
        MoxleyConfig(head_dim=64)
