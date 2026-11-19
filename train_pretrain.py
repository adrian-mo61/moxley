#!/usr/bin/env python
"""Pretrain Moxley-64M on uint16 token shards.

Single-process, single-GPU (or CPU) trainer with:

* AdamW + warmup/cosine learning-rate schedule
* gradient accumulation and gradient clipping
* bf16/fp16 autocast on CUDA
* periodic validation and resumable checkpoints

For multi-GPU training, wrap this script with ``torchrun`` and a DistributedSampler
in a future revision; a 64M model is comfortably trainable on one 24GB GPU.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from moxley.config import MoxleyConfig  # noqa: E402
from moxley.data import TokenDataset, load_metadata  # noqa: E402
from moxley.model import MoxleyForCausalLM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain Moxley-64M")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "moxley_64m.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "bin")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="stop after N optimizer steps (0 = use epochs)",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_dtype(name: str, device: torch.device) -> tuple[torch.dtype, bool]:
    if name == "auto":
        if device.type == "cuda":
            return (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                True,
            )
        return torch.float32, False
    if name == "fp32":
        return torch.float32, False
    if device.type != "cuda":
        raise SystemExit("bf16/fp16 autocast requires a CUDA device")
    return (torch.bfloat16 if name == "bf16" else torch.float16), True


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def learning_rate(step: int, total_steps: int, base_lr: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    if total_steps <= warmup:
        return base_lr
    progress = min(1.0, (step - warmup) / max(total_steps - warmup, 1))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(
    model: MoxleyForCausalLM,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
    max_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_index, (x, y) in enumerate(loader):
        if batch_index >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            loss = model(x, labels=y).loss
        losses.append(float(loss.detach()))
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: MoxleyConfig,
    args: argparse.Namespace,
    step: int,
    epoch: int,
    batch_in_epoch: int,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": {key: value.detach().cpu() for key, value in unwrap(model).state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config.to_dict(),
        "args": vars(args),
        "step": step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "val_loss": val_loss,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        torch.save(payload, handle)
    os.replace(temporary, path)


def make_grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # PyTorch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


def main() -> int:
    args = parse_args()
    if args.seq_len < 2:
        raise SystemExit("--seq-len must be at least 2")
    if args.batch_size < 1 or args.grad_accum < 1:
        raise SystemExit("--batch-size and --grad-accum must be positive")
    if args.epochs < 1:
        raise SystemExit("--epochs must be positive")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    device = resolve_device(args.device)
    amp_dtype, amp_enabled = resolve_dtype(args.dtype, device)
    config = MoxleyConfig.from_pretrained(args.config)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.data_dir / "train.bin"
    val_path = args.data_dir / "val.bin"
    if not train_path.exists():
        raise SystemExit(
            f"{train_path} not found; run python scripts/prepare_data.py first"
        )
    meta = load_metadata(args.data_dir) if (args.data_dir / "meta.json").exists() else {}
    train_dataset = TokenDataset(train_path, args.seq_len)
    if val_path.exists():
        val_dataset = TokenDataset(val_path, args.seq_len)
    else:
        val_dataset = train_dataset
        print("warning: val.bin not found, validating on train.bin", flush=True)

    model = MoxleyForCausalLM(config).to(device)
    model.train()
    parameter_count = model.num_parameters()
    print(
        f"Moxley: {parameter_count:,} parameters ({parameter_count / 1e6:.2f}M), "
        f"device={device}, amp={amp_dtype if amp_enabled else 'fp32'}",
        flush=True,
    )
    print(f"data: {len(train_dataset):,} train blocks, {len(val_dataset):,} val blocks", flush=True)
    if meta:
        stats = meta.get("stats", {})
        total_tokens = stats.get(
            "tokens", stats.get("train_tokens", 0) + stats.get("val_tokens", 0)
        )
        print(f"tokens: {total_tokens:,}", flush=True)

    if args.compile:
        if device.type != "cuda":
            print("warning: torch.compile is usually not useful on CPU; ignoring", flush=True)
        else:
            model = torch.compile(model)
            print("torch.compile enabled", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    scaler = make_grad_scaler(amp_enabled and amp_dtype == torch.float16)

    def make_loader(dataset: TokenDataset, epoch: int, shuffle: bool) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(args.seed + epoch)
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            generator=generator,
            persistent_workers=False,
        )

    eval_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # steps_per_epoch is computed from the shuffled loader length; drop_last=True
    # makes it independent of the epoch.
    probe_loader = make_loader(train_dataset, 0, shuffle=True)
    steps_per_epoch = len(probe_loader) // args.grad_accum
    if steps_per_epoch < 1:
        raise SystemExit(
            f"only {len(probe_loader)} batches available per epoch, but "
            f"grad_accum={args.grad_accum}; add data or lower batch/accum"
        )
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch

    start_epoch, start_batch, global_step, best_val = 0, 0, 0, float("inf")
    checkpoint_path = args.out_dir / "latest.pt"
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        unwrap(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        global_step = int(checkpoint["step"])
        start_epoch = int(checkpoint["epoch"])
        start_batch = int(checkpoint.get("batch_in_epoch", 0))
        best_val = float(checkpoint.get("val_loss", float("inf")))
        if start_batch >= steps_per_epoch * args.grad_accum:
            start_epoch += 1
            start_batch = 0
        print(
            f"resumed from {checkpoint_path} at epoch={start_epoch + 1}, "
            f"step={global_step}, batch={start_batch}",
            flush=True,
        )
    elif args.resume:
        print(f"no checkpoint at {checkpoint_path}; starting from scratch", flush=True)

    log_path = args.out_dir / "train_log.csv"
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    log_mode = "a" if log_path.exists() else "w"
    log_handle = open(log_path, log_mode, encoding="utf-8", newline="")
    log_writer = csv.writer(log_handle)
    if write_header:
        log_writer.writerow(["step", "epoch", "train_loss", "val_loss", "lr", "tokens", "elapsed_s"])

    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
    run_started = time.time()
    tokens_seen = global_step * tokens_per_step
    last_log = time.time()
    last_log_tokens = tokens_seen
    stop = False
    epoch = start_epoch
    epoch_offset = start_batch
    micro_step = start_batch
    loss_accum = 0.0

    for epoch in range(start_epoch, args.epochs):
        loader = make_loader(train_dataset, epoch, shuffle=True)
        epoch_offset = start_batch
        micro_step = 0
        for x, y in itertools.islice(loader, start_batch, None):
            if micro_step >= steps_per_epoch * args.grad_accum:
                break
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop = True
                break

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss = model(x, labels=y).loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at step {global_step + 1}; lower the learning "
                    "rate or inspect the data shard"
                )
            scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()
            loss_accum += float(loss.detach())

            micro_step += 1
            if micro_step % args.grad_accum != 0:
                continue

            lr = learning_rate(global_step, total_steps, args.lr, args.min_lr, args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            tokens_seen += tokens_per_step
            current_loss = loss_accum / args.grad_accum
            loss_accum = 0.0

            if global_step % args.log_interval == 0 or global_step == total_steps:
                now = time.time()
                elapsed = max(now - last_log, 1e-6)
                interval_tokens = tokens_seen - last_log_tokens
                print(
                    f"step {global_step}/{total_steps} | epoch {epoch + 1} | "
                    f"loss {current_loss:.4f} | lr {lr:.2e} | "
                    f"{interval_tokens / elapsed:,.0f} tok/s",
                    flush=True,
                )
                log_writer.writerow(
                    [global_step, epoch + 1, f"{current_loss:.6f}", "", f"{lr:.8f}", tokens_seen, f"{now - run_started:.1f}"]
                )
                log_handle.flush()
                last_log = now
                last_log_tokens = tokens_seen

            if args.eval_interval > 0 and (
                global_step % args.eval_interval == 0 or global_step == total_steps
            ):
                val_loss = evaluate(
                    model,
                    eval_loader,
                    device,
                    amp_dtype,
                    amp_enabled,
                    args.eval_batches,
                )
                best_val = min(best_val, val_loss)
                print(f"  validation loss: {val_loss:.4f} (best {best_val:.4f})", flush=True)
                log_writer.writerow(
                    [global_step, epoch + 1, "", f"{val_loss:.6f}", f"{lr:.8f}", tokens_seen, f"{time.time() - run_started:.1f}"]
                )
                log_handle.flush()

            if args.save_interval > 0 and global_step % args.save_interval == 0:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scaler,
                    config,
                    args,
                    global_step,
                    epoch,
                    epoch_offset + micro_step,
                    best_val,
                )
                print(f"  saved {checkpoint_path}", flush=True)

            if args.max_steps > 0 and global_step >= args.max_steps:
                stop = True
                break

        start_batch = 0
        if stop:
            break

    final_val = evaluate(
        model, eval_loader, device, amp_dtype, amp_enabled, args.eval_batches
    )
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scaler,
        config,
        args,
        global_step,
        epoch,
        epoch_offset + micro_step,
        final_val,
    )
    save_checkpoint(
        args.out_dir / "final.pt",
        model,
        optimizer,
        scaler,
        config,
        args,
        global_step,
        epoch,
        epoch_offset + micro_step,
        final_val,
    )
    unwrap(model).save_pretrained(args.out_dir / "final")
    print(
        f"done: step={global_step}, final validation loss={final_val:.4f}, "
        f"checkpoint={checkpoint_path}",
        flush=True,
    )
    log_writer.writerow(
        [global_step, "final", "", f"{final_val:.6f}", "", tokens_seen, f"{time.time() - run_started:.1f}"]
    )
    log_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
