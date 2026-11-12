"""Contrastive mid-training that adapts Chinese CLIP from one pass to a loop.

The student is ``LoopFuse_partial_loop``: Chinese CLIP weights, with the
middle block run ``num_loops`` times. The first ``warmup_steps`` keep
``num_loops`` at 1, then training switches to the target loop count. The loss is a contrastive loss at every loop. Each pass projects its own
image and text vectors and aligns the paired sample, the same role a golden
token plays for every recurrent step in looped-LM finetuning. The step loss
is the mean of those loop losses.

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
                            picture = Image.open(self._resolve(str(image))).convert("RGB")
                        except (OSError, FileNotFoundError):
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

    files = list_jsonl(Path(args.data))
    dataset = ImageTextStream(files, Path(args.image_root) if args.image_root else None, rank, world, args.seed)
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
            f"mid-train clip={args.clip} files={len(files)} loops={args.num_loops} "
            f"warmup={args.warmup_steps} batch={args.batch_size}x{world} device={device}",
            flush=True,
        )

    start = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = int(checkpoint["step"])

    amp = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    student.train()
    step = start
    running = {"loss": 0.0, "per_loop": []}
    iterator = iter(loader)
    progress = tqdm(total=args.steps, initial=start, disable=not is_main)
    while step < args.steps:
        batch = next(iterator)
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        loops = 1 if step < args.warmup_steps else args.num_loops
        progress_scale = min(1.0, step / max(args.lr_warmup, 1))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * progress_scale * 0.5 * (1.0 + math.cos(math.pi * step / args.steps))

        with torch.autocast(device_type=device.type, dtype=amp, enabled=amp == torch.bfloat16):
            loop_pairs = student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                num_loops=loops,
            )
            step_losses = [
                contrastive_loss(gather_with_grad(image.float()), gather_with_grad(text.float()), module.logit_scale)
                for text, image in loop_pairs
            ]
            loss = torch.stack(step_losses).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()
        with torch.no_grad():
            module.logit_scale.clamp_(0.0, math.log(100.0))

        step += 1
        progress.update(1)
        running["loss"] += float(loss.detach())
        per_loop = running["per_loop"]
        for index, item in enumerate(step_losses):
            if index == len(per_loop):
                per_loop.append(0.0)
            per_loop[index] += float(item.detach())
        if is_main and step % args.log_every == 0:
            parts = " ".join(f"l{index} {value / args.log_every:.4f}" for index, value in enumerate(per_loop))
            print(
                f"step {step}/{args.steps} loops={loops} loss {running['loss'] / args.log_every:.4f} "
                f"{parts} logit_scale {float(module.logit_scale.exp()):.2f}",
                flush=True,
            )
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
    parser.add_argument("--data", required=True, help="A jsonl file, or a directory of jsonl files, with text and image.")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--output", default="outputs/midtrain_partial")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000, help="Steps that still use a single pass.")
    parser.add_argument("--lr-warmup", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
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
    parser.add_argument("--resume", default="")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
