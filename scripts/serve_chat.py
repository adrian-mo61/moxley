#!/usr/bin/env python
"""Minimal streaming chat server for a Moxley SFT model.

Only the Python standard library and PyTorch are required. The browser receives
Server-Sent Events (SSE) from ``POST /api/chat`` and renders tokens as they are
generated.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote, urlparse

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moxley.model import MoxleyForCausalLM  # noqa: E402
from moxley.sft import render_conversation  # noqa: E402
from moxley.tokenizer import BBPETokenizer, SPECIAL_TOKENS  # noqa: E402

WEB_DIR = ROOT / "web"
MODEL_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Moxley streaming chat server")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "moxley-64m-sft-mini" / "final",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "models" / "tokenizer",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument(
        "--system",
        default="你是一个严谨、简洁的中文助手。",
        help="default system prompt; pass an empty string to disable",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def clean_messages(raw: object, default_system: str) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("messages must be a non-empty array")
    messages: list[dict] = []
    for item in raw[-24:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        content = item.get("content")
        if not isinstance(content, str):
            content = str(content or "")
        content = content[:6000]
        if content.strip() or role == "assistant":
            messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("no usable messages")
    if messages[0].get("role") != "system" and default_system:
        messages.insert(0, {"role": "system", "content": default_system})
    return messages


def _safe_decode(tokenizer: BBPETokenizer, ids: list[int]) -> str:
    text = tokenizer.decode(ids, skip_special_tokens=True)
    # Byte-level BPE can leave a partial UTF-8 sequence at the end of a chunk.
    # Withhold a trailing replacement char until the next token completes it.
    if text.endswith("\ufffd"):
        text = text[:-1]
    return text


def stream_chat(
    model: MoxleyForCausalLM,
    tokenizer: BBPETokenizer,
    messages: list[dict],
    device: torch.device,
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Iterator[dict]:
    prompt, _ = render_conversation(messages, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) > max_prompt_tokens:
        prompt_ids = prompt_ids[-max_prompt_tokens:]
    if not prompt_ids:
        raise ValueError("empty prompt")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    think_open = tokenizer.token_to_id("<think>")
    think_close = tokenizer.token_to_id("</think>")
    special_ids = {
        token_id
        for token in SPECIAL_TOKENS
        if (token_id := tokenizer.token_to_id(token)) is not None
    }
    channels: dict[str, list[int]] = {"think": [], "answer": []}
    sent_length = {"think": 0, "answer": 0}
    mode = "answer"
    generated_count = 0

    for token_id in model.generate_stream(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=50,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        repetition_penalty=1.1,
        no_repeat_ngram_size=4,
    ):
        generated_count += 1
        if think_open is not None and token_id == think_open:
            mode = "think"
            continue
        if think_close is not None and token_id == think_close:
            mode = "answer"
            continue
        # Other special tokens are structural and should not appear in chat.
        if token_id in special_ids:
            continue

        channels[mode].append(token_id)
        text = _safe_decode(tokenizer, channels[mode])
        if len(text) > sent_length[mode]:
            delta = text[sent_length[mode] :]
            sent_length[mode] = len(text)
            yield {"type": "token", "channel": mode, "text": delta}

    yield {
        "type": "done",
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": generated_count,
    }


class ChatHandler(BaseHTTPRequestHandler):
    model: MoxleyForCausalLM | None = None
    tokenizer: BBPETokenizer | None = None
    device: torch.device
    max_prompt_tokens: int = 2048
    max_new_tokens: int = 384
    default_system: str = ""
    model_label: str = ""

    server_version = "MoxleyChat/0.1"

    def log_message(self, format: str, *args) -> None:
        # Keep the terminal quiet; print only startup and errors.
        return

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404)
            return
        content = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._serve_file(WEB_DIR / "index.html")
            return
        if path == "/api/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "model": self.model_label,
                    "device": str(self.device),
                },
            )
            return
        if path.startswith("/static/"):
            relative = unquote(path[len("/static/") :])
            target = (WEB_DIR / relative).resolve()
            try:
                target.relative_to(WEB_DIR.resolve())
            except ValueError:
                self.send_error(403)
                return
            self._serve_file(target)
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/chat":
            self.send_error(404)
            return
        if self.model is None or self.tokenizer is None:
            self._send_json(503, {"error": "model not loaded"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2 * 1024 * 1024:
                raise ValueError("invalid request body")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            messages = clean_messages(payload.get("messages"), self.default_system)
            temperature = min(2.0, max(0.0, float(payload.get("temperature", 0.6))))
            top_p = min(1.0, max(0.1, float(payload.get("top_p", 0.9))))
            max_new = min(
                self.max_new_tokens,
                max(1, int(payload.get("max_new_tokens", self.max_new_tokens))),
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def emit(event: dict) -> None:
            data = json.dumps(event, ensure_ascii=False)
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

        try:
            with MODEL_LOCK:
                for event in stream_chat(
                    self.model,
                    self.tokenizer,
                    messages,
                    self.device,
                    self.max_prompt_tokens,
                    max_new,
                    temperature,
                    top_p,
                ):
                    emit(event)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:  # noqa: BLE001 - report failures to the browser
            try:
                emit({"type": "error", "message": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass


def main() -> int:
    args = parse_args()
    if not (WEB_DIR / "index.html").exists():
        raise SystemExit(f"web UI not found: {WEB_DIR / 'index.html'}")
    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")
    if not args.tokenizer.exists():
        raise SystemExit(f"tokenizer not found: {args.tokenizer}")

    device = resolve_device(args.device)
    torch.set_float32_matmul_precision("high")
    tokenizer = BBPETokenizer.from_pretrained(args.tokenizer)
    model = MoxleyForCausalLM.from_pretrained(args.model, map_location="cpu")
    model.to(device).eval()

    ChatHandler.model = model
    ChatHandler.tokenizer = tokenizer
    ChatHandler.device = device
    ChatHandler.max_prompt_tokens = args.max_prompt_tokens
    ChatHandler.max_new_tokens = args.max_new_tokens
    ChatHandler.default_system = args.system
    ChatHandler.model_label = str(args.model)

    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    server.daemon_threads = True
    print(
        f"Moxley chat ready on http://{args.host}:{args.port} "
        f"(device={device}, max_new_tokens={args.max_new_tokens})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
