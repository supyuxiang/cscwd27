from __future__ import annotations

import contextlib
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
from tqdm import tqdm
from transformers import AutoProcessor

from src.modeling_encoder import FullModel
from src.utils import (
    build_dataloaders,
    load_checkpoint,
    move_batch,
    run_epoch_eval,
    set_seed,
)


def build_model(config: dict) -> FullModel:
    set_seed(int(config.get("seed", 42)))
    num_heads_q = int(config.get("num_heads_q", config.get("n_heads", 8)))
    num_heads_kv = int(config.get("num_heads_kv", num_heads_q))
    model = FullModel(
        clip_path=str(config["clip_name"]),
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        dropout_p=float(config.get("dropout", 0.1)),
        num_classes=int(config.get("num_classes", 2)),
    )
    for parameter in model.parameters():
        parameter.requires_grad = True
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable != total:
        raise RuntimeError(f"Full finetune expected every parameter trainable, got {trainable}/{total}.")
    return model


def load_experiment(config: dict, *, distributed: bool, rank: int, world_size: int, is_main: bool):
    set_seed(int(config.get("seed", 42)))
    processor = AutoProcessor.from_pretrained(str(config["clip_name"]), trust_remote_code=True)
    train_loader, val_loader, test_loader, train_sampler = build_dataloaders(
        config,
        processor,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        is_main=is_main,
    )
    model = build_model(config)
    if is_main:
        total = sum(parameter.numel() for parameter in model.parameters())
        print(f"full finetune: {total} trainable parameters, loops={int(config.get('num_loops', 3))}")
    return model, train_loader, val_loader, test_loader, train_sampler


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        return model.module
    return model


def _amp_dtype(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return None, None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, None
    try:
        scaler = torch.amp.GradScaler("cuda")
    except TypeError:
        scaler = torch.cuda.amp.GradScaler()
    return torch.float16, scaler


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_loops: int,
    *,
    amp_dtype,
    scaler,
    max_grad_norm: float,
    accum_steps: int,
    distributed: bool,
    is_main: bool,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_count = 0
    for step, batch in enumerate(tqdm(loader, disable=not is_main, leave=False)):
        batch = move_batch(batch, device)
        sync = (step + 1) % accum_steps == 0 or (step + 1) == len(loader)
        sync_context = model.no_sync() if distributed and not sync else contextlib.nullcontext()
        with sync_context:
            enabled = amp_dtype is not None
            with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=enabled):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    num_loops=num_loops,
                )
                loss = F.cross_entropy(logits, batch["labels"])
            backward_loss = loss / accum_steps
            if scaler is None:
                backward_loss.backward()
            else:
                scaler.scale(backward_loss).backward()
        batch_size = int(batch["labels"].size(0))
        total_loss += float(loss.detach().item()) * batch_size
        total_count += batch_size
        if not sync:
            continue
        if scaler is None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        else:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad(set_to_none=True)

    if distributed:
        packed = torch.tensor([total_loss, total_count], dtype=torch.float64, device=device)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        total_loss = float(packed[0].item())
        total_count = int(packed[1].item())
    return total_loss / max(total_count, 1)


def _broadcast_stop(stop: bool, device: torch.device, distributed: bool) -> bool:
    if not distributed:
        return stop
    flag = torch.tensor([1 if stop else 0], device=device)
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def save_checkpoint(path: Path, model: torch.nn.Module, epoch: int, metrics: dict | None, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in unwrap_model(model).state_dict().items()}
    torch.save({"model": state, "epoch": epoch, "metrics": metrics, "config": config}, path)


def run_training(
    config: dict,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    train_sampler=None,
    *,
    is_main: bool = True,
    distributed: bool = False,
) -> None:
    num_loops = int(config.get("num_loops", 3))
    epochs = int(config["epochs"])
    patience = int(config.get("patience", 0))
    accum_steps = max(int(config.get("grad_accum_steps", 1)), 1)
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    amp_dtype, scaler = _amp_dtype(device, bool(config.get("amp", True)))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    output_dir = Path(config.get("output_dir", "outputs/run"))
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"device={device} amp={amp_dtype} accum={accum_steps} "
            f"lr={config['lr']} epochs={epochs}"
        )

    best_score = -1.0
    stale = 0
    epoch = 0
    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            num_loops,
            amp_dtype=amp_dtype,
            scaler=scaler,
            max_grad_norm=max_grad_norm,
            accum_steps=accum_steps,
            distributed=distributed,
            is_main=is_main,
        )
        metrics = None
        stop = False
        if is_main:
            message = f"epoch {epoch}/{epochs} loss {loss:.4f}"
            if val_loader is not None:
                metrics, report = run_epoch_eval(unwrap_model(model), val_loader, device, num_loops)
                message += f" acc {metrics['accuracy']:.4f} macro_f1 {metrics['macro_f1']:.4f} f1_fake {metrics['f1_fake']:.4f}"
                if metrics["macro_f1"] > best_score:
                    best_score = metrics["macro_f1"]
                    stale = 0
                    save_checkpoint(output_dir / "best.pt", model, epoch, metrics, config)
                    print(message)
                    print(report)
                else:
                    stale += 1
                    if patience > 0:
                        message += f" stale {stale}/{patience}"
                    print(message)
                stop = patience > 0 and stale >= patience
            else:
                print(message)
            save_checkpoint(output_dir / "last.pt", model, epoch, metrics, config)
        if _broadcast_stop(stop, device, distributed):
            if is_main:
                print(f"early stop at epoch {epoch}")
            break

    if is_main and val_loader is None:
        save_checkpoint(output_dir / "best.pt", model, epoch, None, config)
    if is_main and test_loader is not None:
        best_path = output_dir / "best.pt"
        if best_path.is_file():
            checkpoint = load_checkpoint(best_path, device)
            unwrap_model(model).load_state_dict(checkpoint["model"])
        metrics, report = run_epoch_eval(unwrap_model(model), test_loader, device, num_loops)
        print("test", " ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
        print(report)
    if distributed:
        dist.barrier()


def init_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training expects CUDA. Launch with torchrun.")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank, device
