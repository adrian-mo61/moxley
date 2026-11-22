#!/usr/bin/env python
"""Supervised fine-tuning for Moxley-64M on ChatML SFT shards.

The SFT data is pre-tokenized into:

* ``tokens.bin``  uint16 token stream
* ``mask.bin``    uint8, 1 = assistant token contributes to loss
* ``offsets.npy`` int64 sample boundaries
* ``train_indices.npy`` / ``val_indices.npy``

This keeps the expensive JSONL parsing and tokenization out of the training
loop and avoids padding entire 14 GB of conversations on every epoch.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from moxley.config import MoxleyConfig  # noqa: E402
from moxley.data import load_metadata  # noqa: E402
from moxley.model import MoxleyForCausalLM  # noqa: E402
from moxley.sft import SFTBinDataset, collate_sft  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Moxley-64M supervised fine-tuning")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "moxley_64m.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "sft_bin_moxley")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out" / "moxley-64m-sft")
    parser.add_argument(
        "--init-from",
        type=Path,
        default=ROOT / "models" / "moxley-64m-pretrain",
        help="pretrained checkpoint directory or .pt file",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="do not load a pretrained checkpoint (normally not recommended)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="stop after N optimizer steps (0 = use epochs)",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        help="require resuming from out-dir/latest.pt (default: auto-detect)",
    )
    resume_group.add_argument(
        "--fresh",
        action="store_true",
        help="ignore latest.pt and start a new SFT run",
    )
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


def make_grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def load_pretrained(model: MoxleyForCausalLM, path: Path) -> None:
    if path.is_dir():
        path = path / "pytorch_model.bin"
    if not path.exists():
        raise SystemExit(
            f"pretrained checkpoint not found: {path}\n"
            "Pass --init-from /path/to/final, or use --from-scratch for a debug run."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"warning: checkpoint load had {len(missing)} missing and "
            f"{len(unexpected)} unexpected keys",
            flush=True,
        )
        if missing:
            print(f"  missing: {missing[:8]}", flush=True)
        if unexpected:
            print(f"  unexpected: {unexpected[:8]}", flush=True)


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
    loss_sum = 0.0
    token_count = 0
    for batch_index, (x, y) in enumerate(loader):
        if batch_index >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        count = int((y != -100).sum())
        if count == 0:
            continue
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            loss = model(x, labels=y).loss
        loss_sum += float(loss.detach()) * count
        token_count += count
    model.train()
    return loss_sum / token_count if token_count else float("nan")


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
        "stage": "sft",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        torch.save(payload, handle)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
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

    data_dir = args.data_dir
    required = ["tokens.bin", "mask.bin", "offsets.npy", "train_indices.npy"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise SystemExit(
            f"missing SFT shards in {data_dir}: {missing}\n"
            "Run: python scripts/rewrite_sft_identity.py "
            "--input data/sft_t2t.jsonl --tokenizer data/tokenizer "
            "--output-dir data/sft_bin_moxley"
        )

    meta = load_metadata(data_dir) if (data_dir / "meta.json").exists() else {}
    train_dataset = SFTBinDataset(
        data_dir / "tokens.bin",
        data_dir / "mask.bin",
        data_dir / "offsets.npy",
        data_dir / "train_indices.npy",
    )
    val_indices = data_dir / "val_indices.npy"
    if val_indices.exists():
        val_dataset = SFTBinDataset(
            data_dir / "tokens.bin",
            data_dir / "mask.bin",
            data_dir / "offsets.npy",
            val_indices,
        )
    else:
        val_dataset = train_dataset
        print("warning: val_indices.npy not found, validating on train data", flush=True)

    model = MoxleyForCausalLM(config)
    if not args.from_scratch:
        print(f"loading pretrained weights: {args.init_from}", flush=True)
        load_pretrained(model, args.init_from)
    else:
        print("warning: --from-scratch set; SFT starts from random weights", flush=True)
    model.to(device).train()

    params = model.num_parameters()
    print(
        f"Moxley SFT: {params:,} parameters ({params / 1e6:.2f}M), "
        f"device={device}, amp={amp_dtype if amp_enabled else 'fp32'}",
        flush=True,
    )
    print(
        f"data: {len(train_dataset):,} train samples, "
        f"{len(val_dataset):,} val samples",
        flush=True,
    )
    if meta:
        stats = meta.get("stats", {})
        print(
            f"tokens: {stats.get('tokens', 0):,} total, "
            f"{stats.get('train_tokens', 0):,} train, "
            f"{stats.get('val_tokens', 0):,} val; max_len={meta.get('max_len')}",
            flush=True,
        )

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

    def make_loader(
        dataset: SFTBinDataset,
        epoch: int,
        shuffle: bool,
        start_batch: int = 0,
    ) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(args.seed + epoch)
        if not shuffle:
            return DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_sft,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=False,
            )
        # Build the epoch permutation once, then skip whole batches by index.
        # Resuming used to walk DataLoader through every skipped batch, which
        # decoded millions of samples before the first optimizer step.
        order = torch.randperm(len(dataset), generator=generator).tolist()
        start = max(0, start_batch) * args.batch_size
        order = order[start:]
        usable = (len(order) // args.batch_size) * args.batch_size
        order = order[:usable]
        if not order:
            raise ValueError(
                f"resume start_batch={start_batch} is beyond the epoch "
                f"({len(dataset)} samples)"
            )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=order,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_sft,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
        )

    eval_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_sft,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    probe_loader = make_loader(train_dataset, 0, shuffle=True)
    steps_per_epoch = len(probe_loader) // args.grad_accum
    if steps_per_epoch < 1:
        raise SystemExit(
            f"only {len(probe_loader)} SFT batches available per epoch, but "
            f"grad_accum={args.grad_accum}; add data or lower batch/accum"
        )
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch

    start_epoch, start_batch, global_step, best_val = 0, 0, 0, float("inf")
    checkpoint_path = args.out_dir / "latest.pt"
    resumed = False
    if checkpoint_path.exists() and not args.fresh:
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
        resumed = True
        print(
            f"resumed from {checkpoint_path} at epoch={start_epoch + 1}, "
            f"step={global_step}, batch={start_batch}",
            flush=True,
        )
    elif args.resume:
        raise SystemExit(f"--resume requested but {checkpoint_path} does not exist")
    elif args.fresh and checkpoint_path.exists():
        print(f"--fresh: ignoring {checkpoint_path}", flush=True)

    log_path = args.out_dir / "train_log.csv"
    write_header = not resumed or not log_path.exists() or log_path.stat().st_size == 0
    log_mode = "a" if resumed and log_path.exists() else "w"
    log_handle = open(log_path, log_mode, encoding="utf-8", newline="")
    log_writer = csv.writer(log_handle)
    if write_header:
        log_writer.writerow(
            ["step", "epoch", "train_loss", "val_loss", "lr", "supervised_tokens", "elapsed_s"]
        )

    run_started = time.time()
    tokens_seen = 0
    last_log = time.time()
    last_log_tokens = 0
    stop = False
    epoch = start_epoch
    epoch_offset = start_batch
    micro_step = start_batch
    loss_sum = 0.0
    loss_tokens = 0

    progress = None
    if not args.no_progress:
        if tqdm is None:
            print(
                "warning: tqdm is not installed; progress bar disabled. "
                "Install it with: pip install tqdm",
                flush=True,
            )
        else:
            progress = tqdm(
                total=max(total_steps, global_step),
                initial=min(global_step, max(total_steps, global_step)),
                desc="Moxley-SFT",
                unit="step",
                dynamic_ncols=True,
                leave=True,
                file=sys.stderr,
            )

    def emit(message: str) -> None:
        if progress is not None:
            progress.write(message)
        else:
            print(message, flush=True)

    for epoch in range(start_epoch, args.epochs):
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        loader = make_loader(
            train_dataset,
            epoch,
            shuffle=True,
            start_batch=epoch_start_batch,
        )
        epoch_offset = epoch_start_batch
        micro_step = 0
        for x, y in loader:
            if micro_step >= steps_per_epoch * args.grad_accum:
                break
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop = True
                break

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            count = int((y != -100).sum())
            if count == 0:
                micro_step += 1
                continue
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss = model(x, labels=y).loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite SFT loss at step {global_step + 1}; lower the "
                    "learning rate or inspect the data"
                )

            scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()
            loss_sum += float(loss.detach()) * count
            loss_tokens += count
            tokens_seen += count

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
            if progress is not None:
                progress.update(1)

            current_loss = loss_sum / max(loss_tokens, 1)
            loss_sum = 0.0
            loss_tokens = 0

            if global_step % args.log_interval == 0 or global_step == total_steps:
                now = time.time()
                elapsed = max(now - last_log, 1e-6)
                interval_tokens = tokens_seen - last_log_tokens
                emit(
                    f"step {global_step}/{total_steps} | epoch {epoch + 1} | "
                    f"loss {current_loss:.4f} | lr {lr:.2e} | "
                    f"{interval_tokens / elapsed:,.0f} supervised tok/s"
                )
                if progress is not None:
                    progress.set_postfix_str(
                        f"loss={current_loss:.4f} lr={lr:.1e} "
                        f"tok/s={interval_tokens / elapsed:,.0f}"
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
                emit(f"  validation loss: {val_loss:.4f} (best {best_val:.4f})")
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
                emit(f"  saved {checkpoint_path}")

            if args.max_steps > 0 and global_step >= args.max_steps:
                stop = True
                break

        start_batch = 0
        if stop:
            break

    if progress is not None:
        progress.close()
        progress = None

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
