"""Moxley: a compact 64M-parameter language model.

The package intentionally keeps the model, tokenizer and data pipeline small and
readable. See ``README.md`` for the full training recipe.
"""

from .config import MoxleyConfig
from .model import MoxleyForCausalLM

__all__ = ["MoxleyConfig", "MoxleyForCausalLM", "__version__"]
__version__ = "0.1.0"
