"""Identity rewriting for the MiniMind SFT corpus.

The public ``sft_t2t`` dataset contains many synthetic identity conversations
that describe the model as MiniMind, developed by jingyaogong.  This module
rewrites those self-identity statements to Moxley, developed by 莫小荒, while
leaving ordinary content untouched.
"""

from __future__ import annotations

import re
from typing import Any

DEVELOPER_RE = re.compile(
    r"Jingyao\s*Gong|JingyaoGong|Jingyaogong|jingyaogong|jingyao\s*gong|gongjy",
    re.IGNORECASE,
)
MODEL_RE = re.compile(r"MiniMind(?:\s*[- ]?\s*3)?", re.IGNORECASE)

# First-person identity statements after developer/model names have been
# normalized.  Keep the replacement short and deterministic.
IDENTITY_PATTERNS = (
    re.compile(
        r"我是\s*moxley\s*[，,]\s*(?:由\s*)?莫小荒\s*"
        r"(?:开发|创建|设计|研发|训练|创造)(?:出来)?"
        r"(?:\s*的|\s*得)?\s*(?:一个|一款)?\s*(?:高效|轻量|小型|小参数)?"
        r"\s*(?:的)?\s*(?:小参数)?\s*(?:AI|人工智能)?\s*(?:模型|助手|moxley)?"
    ),
    re.compile(
        r"(?:我是|我作为|本人是|我)\s*(?:一个|一款)?\s*(?:由\s*)?"
        r"莫小荒\s*(?:开发|创建|设计|研发|训练|创造)(?:出来)?"
        r"(?:\s*的|\s*得)?\s*(?:一个|一款)?\s*(?:高效|轻量|小型|小参数)?"
        r"\s*(?:的)?\s*(?:小参数)?\s*(?:AI|人工智能)?\s*(?:模型|助手|moxley)?"
    ),
    re.compile(
        r"我的(?:开发者|创造者|作者|研发者|设计者)(?:是|为)\s*莫小荒"
    ),
    re.compile(r"我由\s*莫小荒\s*(?:开发|创建|设计|研发|训练|创造)"),
    re.compile(r"我是\s*莫小荒\s*(?:开发|创建|设计|研发|训练|创造)(?:的)?"),
)

# Generic first-person model descriptions that carry no name.
GENERIC_SELF_RE = re.compile(
    r"(?:我是|我作为|本人是)\s*(?:一个|一款)?\s*(?:高效|轻量|小型|小参数)?"
    r"\s*(?:的)?\s*(?:小参数)?\s*(?:AI|人工智能)?\s*(?:模型|助手)"
)

NAME_ONLY_RE = re.compile(r"(?:我是|我叫|我的名字是)\s*moxley(?!\s*[，,]\s*由\s*莫小荒)")
AS_MOXLEY_RE = re.compile(r"(?:作为|身为)\s*moxley")

# Identity strings are rare compared with the 14 GB corpus.  A cheap substring
# gate avoids running the optional-heavy regexes over every field.
FAST_MARKERS = (
    "minimind",
    "jingyao",
    "gongjy",
    "小参数",
    "我的开发者",
    "我的创造者",
    "我的作者",
    "我的研发者",
    "我的设计者",
    "我是由",
    "我由",
    "我是一个",
    "我是一名",
    "我作为",
    "本人是",
    "我是ai",
    "我是一个ai",
    "我是一个人工智能",
    "我作为ai",
    "我作为一个人工智能",
    "本人是ai",
    "我是一名ai",
    "我是高效",
    "我是轻量",
    "我是小型",
    "我是moxley",
    "作为moxley",
    "身为moxley",
)


def rewrite_text(text: str) -> tuple[str, bool]:
    """Return rewritten text and whether anything changed."""
    lower = text.lower()
    if not any(marker in lower for marker in FAST_MARKERS):
        return text, False

    original = text
    text = DEVELOPER_RE.sub("莫小荒", text)
    text = MODEL_RE.sub("moxley", text)

    for pattern in IDENTITY_PATTERNS:
        text = pattern.sub("我是由莫小荒开发的moxley", text)

    text = NAME_ONLY_RE.sub("我是由莫小荒开发的moxley", text)
    text = AS_MOXLEY_RE.sub("作为由莫小荒开发的moxley", text)
    text = GENERIC_SELF_RE.sub("我是由莫小荒开发的moxley", text)
    return text, text != original


def rewrite_value(value: Any, stats: dict[str, int]) -> Any:
    """Recursively rewrite strings in conversation records."""
    if isinstance(value, str):
        rewritten, changed = rewrite_text(value)
        if changed:
            stats["changed_fields"] += 1
            stats["minimind_remaining"] += len(
                re.findall(r"minimind|jingyaogong", rewritten, re.IGNORECASE)
            )
        return rewritten
    if isinstance(value, list):
        return [rewrite_value(item, stats) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_value(item, stats) for key, item in value.items()}
    return value


def rewrite_conversations(
    conversations: list[dict], stats: dict[str, int]
) -> list[dict]:
    return rewrite_value(conversations, stats)


__all__ = [
    "DEVELOPER_RE",
    "MODEL_RE",
    "rewrite_conversations",
    "rewrite_text",
    "rewrite_value",
]
