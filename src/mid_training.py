"""Contrastive mid-training that adapts Chinese CLIP from one pass to a loop.

The student is ``LoopFuse_partial_loop``: Chinese CLIP weights, with the
middle block run ``num_loops`` times. Each pass projects its own image and
text vectors. The step loss is the contrastive loss of the last loop only.
Earlier loops are logged and still run, because the next pass reads their
hidden state, but they are not in the loss. Keep ``warmup_steps`` at 0 so
``num_loops`` does not drop during training.

The checkpoint stores the student encoders, projections and ``logit_scale``.
A later fine-tune can load those tensors into ``LoopFuse_partial_loop``.

Data is a jsonl file or a directory of them. Each row needs ``text`` and
``image``. Launch with ``torchrun`` for more than one GPU.

Example::

    torchrun --nproc_per_node=4 -m src.mid_training \
        --clip /data3/yxfeng/models/clip/chinese-clip-vit-base-patch16 \
        --data /path/to/image_text_jsonl \
        --output outputs/midtrain_partial \
        --steps 100000 --batch-size 64 --num-loops 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from tqdm import tqdm
from transformers import ChineseCLIPModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.imagenet_zh import ImageNetZhStream
from src.modeling_loopfuse_chinese import LoopFuse_partial_loop


def contrastive_loss(image_embeds: torch.Tensor, text_embeds: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    image = F.normalize(image_embeds, dim=-1)
    text = F.normalize(text_embeds, dim=-1)
    logits = logit_scale.exp().clamp(max=100.0) * image @ text.T
    target = torch.arange(image.size(0), device=image.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


def gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather features. Only this rank's slice keeps a gradient."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    world = dist.get_world_size()
    rank = dist.get_rank()
    pieces = [torch.zeros_like(tensor) for _ in range(world)]
    dist.all_gather(pieces, tensor.detach())
    pieces[rank] = tensor
    return torch.cat(pieces, dim=0)


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    pieces = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(pieces, tensor.contiguous())
    return torch.cat(pieces, dim=0)


def contrastive_chunk_loss(
    image_q: torch.Tensor,
    text_q: torch.Tensor,
    image_all: torch.Tensor,
    text_all: torch.Tensor,
    logit_scale: torch.Tensor,
    query_start: int,
) -> torch.Tensor:
    """InfoNCE for one chunk of queries against the full global batch."""
    image_q = F.normalize(image_q, dim=-1)
    text_q = F.normalize(text_q, dim=-1)
    image_all = F.normalize(image_all, dim=-1)
    text_all = F.normalize(text_all, dim=-1)
    scale = logit_scale.exp().clamp(max=100.0)
    target = torch.arange(image_q.size(0), device=image_q.device) + query_start
    image_logits = scale * image_q @ text_all.T
    text_logits = scale * text_q @ image_all.T
    return (F.cross_entropy(image_logits, target) + F.cross_entropy(text_logits, target)) / 2


class LoopStudent(torch.nn.Module):
    """Partial-loop towers plus the pretrained projections and logit scale."""

    def __init__(self, clip_path: str, num_heads_q: int, num_heads_kv: int, dropout_p: float):
        super().__init__()
        bundle = LoopFuse_partial_loop(clip_path, num_heads_q, num_heads_kv, dropout_p, num_classes=2)
        self.text_encoder = bundle.text_encoder
        self.vision_encoder = bundle.vision_encoder
        self.text_projection = bundle.text_projection
        self.visual_projection = bundle.visual_projection
        self.processor = bundle.processor
        self.logit_scale = torch.nn.Parameter(torch.tensor(2.6592))
        del bundle

    def forward(self, input_ids, attention_mask, pixel_values, num_loops: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        text_loops = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_loops=num_loops,
            return_each_loop=True,
        )
        vision_loops = self.vision_encoder(pixel_values=pixel_values, num_loops=num_loops, return_each_loop=True)
        return [
            (self.text_projection(text), self.visual_projection(image))
            for text, image in zip(text_loops, vision_loops)
        ]


def list_jsonl(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No jsonl files under {path}.")
    return files


class ImageTextStream(IterableDataset):
    """Cycle jsonl shards. Each rank and worker reads a disjoint set of lines."""

    def __init__(self, files: list[Path], image_root: Path | None, rank: int, world: int, seed: int):
        super().__init__()
        self.files = files
        self.image_root = image_root
        self.rank = rank
        self.world = world
        self.seed = seed

    def _resolve(self, image: str) -> Path:
        path = Path(image)
        if path.is_file():
            return path
        if self.image_root is not None:
            candidate = self.image_root / image
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(image)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        stride = self.world * workers
        offset = self.rank * workers + worker_id
        rng = random.Random(self.seed + offset)
        files = list(self.files)
        while True:
            rng.shuffle(files)
            for path in files:
                with path.open(encoding="utf-8") as handle:
                    for index, line in enumerate(handle):
                        if (index % stride) != offset or not line.strip():
                            continue
                        row = json.loads(line)
                        text = str(row.get("text") or "").strip()
                        image = row.get("image")
                        if not text or not image:
                            continue
                        try:
                            picture = Image.open(self._resolve(str(image)))
                            picture.load()
                            picture = picture.convert("RGB")
                        except (OSError, SyntaxError, ValueError):
                            continue
                        yield {"text": text, "image": picture}


def collate(processor, max_length: int):
    """Random-crop each image to 224 before the CLIP processor normalizes it."""
    crop = RandomResizedCrop(224, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC)

    def collate_fn(batch: list[dict]) -> dict:
        images = [crop(item["image"]) for item in batch]
        encoded = processor(
            text=[item["text"] for item in batch],
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            do_resize=False,
        )
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "pixel_values": encoded["pixel_values"]}

    return collate_fn


def text_gate_bias(module: torch.nn.Module) -> float | None:
    for name, parameter in module.named_parameters():
        if "gate" in name and name.endswith("bias") and parameter.numel() == 1:
            return float(parameter.detach())
    return None


def append_metrics(path: Path, step: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write((json.dumps({"step": step, "metrics": metrics}, ensure_ascii=False) + "\n").encode())
        handle.flush()


def save_checkpoint(path: Path, student: torch.nn.Module, optimizer, step: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    module = student.module if isinstance(student, DDP) else student
    torch.save(
        {
            "model": {key: value.detach().cpu() for key, value in module.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "step": step,
            "args": vars(args),
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
        world = dist.get_world_size()
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world = 1
    is_main = rank == 0

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    student = LoopStudent(args.clip, args.num_heads_q, args.num_heads_kv, args.dropout).to(device)
    pretrained = ChineseCLIPModel.from_pretrained(args.clip, trust_remote_code=True)
    with torch.no_grad():
        student.logit_scale.copy_(pretrained.logit_scale.detach())
    del pretrained

    if distributed:
        student = DDP(student, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, gradient_as_bucket_view=False, find_unused_parameters=False)

    decay, no_decay = [], []
    module = student.module if isinstance(student, DDP) else student
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias") or "logit_scale" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
    )

    if args.data_format == "imagenet-zh":
        dataset = ImageNetZhStream(Path(args.data), Path(args.label_file), rank, world, args.seed)
        data_note = f"imagenet-zh images={len(dataset.samples)}"
    else:
        files = list_jsonl(Path(args.data))
        dataset = ImageTextStream(files, Path(args.image_root) if args.image_root else None, rank, world, args.seed)
        data_note = f"files={len(files)}"
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate(module.processor, args.max_length),
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    output = Path(args.output)
    if is_main:
        output.mkdir(parents=True, exist_ok=True)
        print(
            f"mid-train clip={args.clip} {data_note} loops={args.num_loops} "
            f"warmup={args.warmup_steps} loss=last "
            f"batch={args.batch_size}x{world}x{args.grad_accum} global_batch={args.batch_size * world * args.grad_accum} "
            f"lr={args.lr} device={device}",
            flush=True,
        )

    if args.resume and args.init:
        raise SystemExit("Use either --resume or --init, not both.")
    start = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = int(checkpoint["step"])
    elif args.init:
        checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
        module.load_state_dict(checkpoint["model"])
        if is_main:
            print(f"init weights from {args.init} step {checkpoint.get('step')}", flush=True)
    metrics_path = output / "metrics.jsonl"
    if is_main and start == 0:
        metrics_path.write_bytes(b"")

    amp = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    student.train()
    step = start
    running = {"loss": 0.0, "per_loop": []}
    iterator = iter(loader)
    progress = tqdm(total=args.steps, initial=start, disable=not is_main)
    no_sync = student.no_sync if isinstance(student, DDP) else nullcontext
    while step < args.steps:
        loops = 1 if step < args.warmup_steps else args.num_loops
        progress_scale = min(1.0, step / max(args.lr_warmup, 1))
        used_lr = args.lr * progress_scale * 0.5 * (1.0 + math.cos(math.pi * step / args.steps))
        for group in optimizer.param_groups:
            group["lr"] = used_lr
        micro_batches = []
        for _ in range(args.grad_accum):
            batch = next(iterator)
            micro_batches.append({key: value.to(device, non_blocking=True) for key, value in batch.items()})

        if args.grad_accum == 1:
            batch = micro_batches[0]
            with torch.autocast(device_type=device.type, dtype=amp, enabled=amp == torch.bfloat16):
                loop_pairs = student(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    num_loops=loops,
                )
                logged = []
                supervised = None
                last = len(loop_pairs) - 1
                for index, (text, image) in enumerate(loop_pairs):
                    item = contrastive_loss(
                        gather_with_grad(image.float()),
                        gather_with_grad(text.float()),
                        module.logit_scale,
                    )
                    logged.append(item.detach())
                    if index == last:
                        supervised = item
                if supervised is None:
                    raise RuntimeError(f"last-loop loss needs a readout, got {len(loop_pairs)} at step {step}.")
                loss = supervised
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
        else:
            cached_image = []
            cached_text = []
            per_loop_local: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
            with no_sync():
                with torch.no_grad():
                    for batch in micro_batches:
                        with torch.autocast(device_type=device.type, dtype=amp, enabled=amp == torch.bfloat16):
                            loop_pairs = student(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                pixel_values=batch["pixel_values"],
                                num_loops=loops,
                            )
                        if not per_loop_local:
                            per_loop_local = [[] for _ in loop_pairs]
                        for index, (text, image) in enumerate(loop_pairs):
                            per_loop_local[index].append((text.float(), image.float()))
                        text, image = loop_pairs[-1]
                        cached_text.append(text.float())
                        cached_image.append(image.float())
            local_image = torch.cat(cached_image, dim=0)
            local_text = torch.cat(cached_text, dim=0)
            all_image = all_gather(local_image)
            all_text = all_gather(local_text)
            micro = micro_batches[0]["input_ids"].size(0)
            local_count = local_image.size(0)
            rank_offset = rank * local_count
            logged = []
            with torch.no_grad():
                for index, chunks in enumerate(per_loop_local):
                    if index == len(per_loop_local) - 1:
                        continue
                    chunk_text = torch.cat([text for text, _ in chunks], dim=0)
                    chunk_image = torch.cat([image for _, image in chunks], dim=0)
                    logged.append(
                        contrastive_loss(all_gather(chunk_image), all_gather(chunk_text), module.logit_scale).detach()
                    )
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for chunk_index, batch in enumerate(micro_batches):
                sync = nullcontext() if chunk_index == args.grad_accum - 1 else no_sync()
                with sync:
                    with torch.autocast(device_type=device.type, dtype=amp, enabled=amp == torch.bfloat16):
                        loop_pairs = student(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            num_loops=loops,
                        )
                        text, image = loop_pairs[-1]
                        query_start = rank_offset + chunk_index * micro
                        image_all = all_image.detach().clone()
                        text_all = all_text.detach().clone()
                        image_all[query_start : query_start + micro] = image.float()
                        text_all[query_start : query_start + micro] = text.float()
                        chunk_loss = contrastive_chunk_loss(
                            image.float(),
                            text.float(),
                            image_all,
                            text_all,
                            module.logit_scale,
                            query_start,
                        )
                    (chunk_loss / args.grad_accum).backward()
                loss_value += float(chunk_loss.detach()) / args.grad_accum
            loss = torch.tensor(loss_value, device=device)
            logged.append(loss.detach())
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()
        with torch.no_grad():
            module.logit_scale.clamp_(0.0, math.log(100.0))

        step += 1
        progress.update(1)
        running["loss"] += float(loss.detach())
        per_loop = running["per_loop"]
        for index, item in enumerate(logged):
            if index == len(per_loop):
                per_loop.append(0.0)
            per_loop[index] += float(item)
        if is_main and step % args.log_every == 0:
            parts = " ".join(f"l{index} {value / args.log_every:.4f}" for index, value in enumerate(per_loop))
            mean_loss = running["loss"] / args.log_every
            print(
                f"step {step}/{args.steps} loops={loops} loss {mean_loss:.4f} "
                f"{parts} logit_scale {float(module.logit_scale.exp()):.2f}",
                flush=True,
            )
            metrics = {
                "train/loss": mean_loss,
                "train/lr": used_lr,
                "train/logit_scale": float(module.logit_scale.exp()),
                "train/loops": float(loops),
                "train/global_batch": float(args.batch_size * world * args.grad_accum),
            }
            for index, value in enumerate(per_loop):
                metrics[f"train/l{index}"] = value / args.log_every
            bias = text_gate_bias(module)
            if bias is not None:
                metrics["train/gate_bias"] = bias
            append_metrics(metrics_path, step, metrics)
            running = {"loss": 0.0, "per_loop": []}
        if is_main and step % args.save_every == 0:
            save_checkpoint(output / f"step_{step}.pt", student, optimizer, step, args)
            save_checkpoint(output / "latest.pt", student, optimizer, step, args)
    if is_main:
        save_checkpoint(output / "latest.pt", student, optimizer, step, args)
        print(f"saved {output / 'latest.pt'}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mid-train Chinese CLIP from one pass to a partial loop.")
    parser.add_argument("--clip", default="/data3/yxfeng/models/clip/chinese-clip-vit-base-patch16")
    parser.add_argument("--data", required=True, help="Jsonl file or directory, or the ImageNet train root when --data-format=imagenet-zh.")
    parser.add_argument("--data-format", default="jsonl", choices=("jsonl", "imagenet-zh"))
    parser.add_argument("--label-file", default="/data3/yxfeng/datasets/imagenet/label_cn.txt")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--output", default="outputs/midtrain_partial")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000, help="Steps that still use a single pass.")
    parser.add_argument("--lr-warmup", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=1, help="Micro-batches per update. Global batch is batch-size × GPUs × grad-accum.")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-loops", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--distill-weight", type=float, default=10.0, help="Unused. Kept so older launch scripts still parse.")
    parser.add_argument("--distill-weight-end", type=float, default=0.1, help="Unused. Kept so older launch scripts still parse.")
    parser.add_argument("--num-heads-q", type=int, default=8)
    parser.add_argument("--num-heads-kv", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default="", help="Resume model, optimizer, and step from this checkpoint.")
    parser.add_argument("--init", default="", help="Load model weights only. Optimizer and step stay fresh.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
