"""Small image-text retrieval probe: vanilla Chinese CLIP vs the loop mid-train."""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import ChineseCLIPModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mid_training import LoopStudent


def sample_pairs(jsonl_dir: Path, n: int, seed: int, max_chars: int) -> list[dict]:
    rng = random.Random(seed)
    files = sorted(jsonl_dir.glob("laion-*.jsonl"))
    rng.shuffle(files)
    kept: list[dict] = []
    seen_images: set[str] = set()
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if len(kept) >= n:
                    return kept
                if not line.strip():
                    continue
                row = json.loads(line)
                text = str(row.get("text") or "").strip()
                image = str(row.get("image") or "")
                chars = int(row.get("chars") or len(text))
                if not text or not image or image in seen_images or not (20 <= chars <= max_chars):
                    continue
                if not Path(image).is_file():
                    continue
                seen_images.add(image)
                kept.append({"text": text, "image": image})
    if len(kept) < n:
        raise RuntimeError(f"only found {len(kept)} usable pairs, wanted {n}")
    return kept


def collate(processor, pairs: list[dict], max_length: int) -> dict:
    images = []
    for item in pairs:
        with Image.open(item["image"]) as picture:
            images.append(picture.convert("RGB"))
    encoded = processor(
        text=[item["text"] for item in pairs],
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "pixel_values": encoded["pixel_values"],
    }


@torch.inference_mode()
def encode_vanilla(model, processor, pairs: list[dict], batch_size: int, max_length: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    images, texts = [], []
    for start in tqdm(range(0, len(pairs), batch_size), desc="vanilla"):
        batch = collate(processor, pairs[start : start + batch_size], max_length)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch, return_loss=False)
        images.append(F.normalize(outputs.image_embeds.float(), dim=-1).cpu())
        texts.append(F.normalize(outputs.text_embeds.float(), dim=-1).cpu())
    return torch.cat(images), torch.cat(texts)


@torch.inference_mode()
def encode_student(student, processor, pairs: list[dict], batch_size: int, max_length: int, device, num_loops: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    buckets: list[tuple[list[torch.Tensor], list[torch.Tensor]]] | None = None
    for start in tqdm(range(0, len(pairs), batch_size), desc="midtrain"):
        batch = collate(processor, pairs[start : start + batch_size], max_length)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loops = student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                num_loops=num_loops,
            )
        if buckets is None:
            buckets = [([], []) for _ in loops]
        for index, (text, image) in enumerate(loops):
            buckets[index][0].append(F.normalize(image.float(), dim=-1).cpu())
            buckets[index][1].append(F.normalize(text.float(), dim=-1).cpu())
    assert buckets is not None
    return [(torch.cat(images), torch.cat(texts)) for images, texts in buckets]


def load_flickr(root: Path) -> tuple[list[Image.Image], list[str], list[int], list[list[int]]]:
    text_rows = [json.loads(line) for line in (root / "test_texts.jsonl").open(encoding="utf-8") if line.strip()]
    images: dict[str, Image.Image] = {}
    with (root / "test_imgs.tsv").open(encoding="utf-8") as handle:
        for line in handle:
            image_id, payload = line.rstrip("\n").split("\t", 1)
            raw = base64.b64decode(payload)
            with Image.open(io.BytesIO(raw)) as picture:
                images[image_id] = picture.convert("RGB")
    image_ids = list(images)
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    texts = [row["text"] for row in text_rows]
    text_to_image = [image_index[str(row["image_ids"][0])] for row in text_rows]
    image_to_texts: list[list[int]] = [[] for _ in image_ids]
    for text_index, image_pos in enumerate(text_to_image):
        image_to_texts[image_pos].append(text_index)
    return [images[image_id] for image_id in image_ids], texts, text_to_image, image_to_texts


def batched_pixels(processor, pictures: list[Image.Image]) -> torch.Tensor:
    encoded = processor(images=pictures, return_tensors="pt")
    return encoded["pixel_values"]


def batched_text(processor, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    encoded = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


@torch.inference_mode()
def embed_images_vanilla(model, processor, pictures, batch_size, device) -> torch.Tensor:
    out = []
    for start in tqdm(range(0, len(pictures), batch_size), desc="vanilla images"):
        pixels = batched_pixels(processor, pictures[start : start + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = model.get_image_features(pixel_values=pixels).pooler_output
        out.append(F.normalize(features.float(), dim=-1).cpu())
    return torch.cat(out)


@torch.inference_mode()
def embed_texts_vanilla(model, processor, texts, batch_size, max_length, device) -> torch.Tensor:
    out = []
    for start in tqdm(range(0, len(texts), batch_size), desc="vanilla texts"):
        batch = batched_text(processor, texts[start : start + batch_size], max_length)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = model.get_text_features(**batch).pooler_output
        out.append(F.normalize(features.float(), dim=-1).cpu())
    return torch.cat(out)


@torch.inference_mode()
def embed_images_student(student, processor, pictures, batch_size, device, num_loops: int) -> list[torch.Tensor]:
    buckets: list[list[torch.Tensor]] | None = None
    for start in tqdm(range(0, len(pictures), batch_size), desc="midtrain images"):
        pixels = batched_pixels(processor, pictures[start : start + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loops = student.vision_encoder(pixel_values=pixels, num_loops=num_loops, return_each_loop=True)
            projected = [student.visual_projection(state) for state in loops]
        if buckets is None:
            buckets = [[] for _ in projected]
        for index, features in enumerate(projected):
            buckets[index].append(F.normalize(features.float(), dim=-1).cpu())
    assert buckets is not None
    return [torch.cat(parts) for parts in buckets]


@torch.inference_mode()
def embed_texts_student(student, processor, texts, batch_size, max_length, device, num_loops: int) -> list[torch.Tensor]:
    buckets: list[list[torch.Tensor]] | None = None
    for start in tqdm(range(0, len(texts), batch_size), desc="midtrain texts"):
        batch = batched_text(processor, texts[start : start + batch_size], max_length)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loops = student.text_encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                num_loops=num_loops,
                return_each_loop=True,
            )
            projected = [student.text_projection(state) for state in loops]
        if buckets is None:
            buckets = [[] for _ in projected]
        for index, features in enumerate(projected):
            buckets[index].append(F.normalize(features.float(), dim=-1).cpu())
    assert buckets is not None
    return [torch.cat(parts) for parts in buckets]


def recall_retrieval(images: torch.Tensor, texts: torch.Tensor, text_to_image: list[int], image_to_texts: list[list[int]]) -> tuple[str, str]:
    similarity = images @ texts.T
    image_hits = []
    for image_index, gold_texts in enumerate(image_to_texts):
        order = torch.argsort(similarity[image_index], descending=True)
        ranks = [(order == gold).nonzero(as_tuple=False)[0].item() for gold in gold_texts]
        image_hits.append(min(ranks))
    text_hits = []
    for text_index, image_index in enumerate(text_to_image):
        order = torch.argsort(similarity[:, text_index], descending=True)
        text_hits.append((order == image_index).nonzero(as_tuple=False)[0].item())
    image_hits_t = torch.tensor(image_hits)
    text_hits_t = torch.tensor(text_hits)
    def pack(ranks: torch.Tensor) -> str:
        return "/".join(f"{float((ranks < k).float().mean()):.3f}" for k in (1, 5, 10))
    return pack(image_hits_t), pack(text_hits_t)


def recall_at(similarity: torch.Tensor, ks: tuple[int, ...]) -> dict[str, float]:
    gold = similarity.diag().unsqueeze(1)
    ranks = (similarity >= gold).sum(dim=1)
    return {f"R@{k}": float((ranks <= k).float().mean()) for k in ks}


def report(name: str, images: torch.Tensor, texts: torch.Tensor) -> None:
    similarity = images @ texts.T
    image_to_text = recall_at(similarity, (1, 5, 10))
    text_to_image = recall_at(similarity.T, (1, 5, 10))
    print(
        f"{name:16} "
        f"I2T {image_to_text['R@1']:.3f}/{image_to_text['R@5']:.3f}/{image_to_text['R@10']:.3f} "
        f"T2I {text_to_image['R@1']:.3f}/{text_to_image['R@5']:.3f}/{text_to_image['R@10']:.3f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default="/data3/yxfeng/models/clip/chinese-clip-vit-base-patch16")
    parser.add_argument("--checkpoint", default="/data3/yxfeng/experiments/cnclip_midtrain_l2/step_5000.pt")
    parser.add_argument("--flickr", default="/data3/yxfeng/datasets/flickr30k-cn/Flickr30k-CN")
    parser.add_argument("--jsonl", default="/data3/yxfeng/datasets/cnclip_pretrain/jsonl")
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-loops", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda")
    pictures, texts, text_to_image, image_to_texts = load_flickr(Path(args.flickr))
    print(f"flickr30k-cn test images={len(pictures)} texts={len(texts)}", flush=True)

    vanilla = ChineseCLIPModel.from_pretrained(args.clip, trust_remote_code=True).to(device).eval()
    processor = vanilla.processor if hasattr(vanilla, "processor") else None
    if processor is None:
        from transformers import ChineseCLIPProcessor

        processor = ChineseCLIPProcessor.from_pretrained(args.clip, trust_remote_code=True)
    image_v = embed_images_vanilla(vanilla, processor, pictures, args.batch_size, device)
    text_v = embed_texts_vanilla(vanilla, processor, texts, args.batch_size, args.max_length, device)
    image_to_text, text_to_image_score = recall_retrieval(image_v, text_v, text_to_image, image_to_texts)
    print(f"{'vanilla':16} I2T {image_to_text} T2I {text_to_image_score}", flush=True)
    del vanilla
    torch.cuda.empty_cache()

    student = LoopStudent(args.clip, num_heads_q=8, num_heads_kv=4, dropout_p=0.0).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    student.load_state_dict(checkpoint["model"])
    print(f"checkpoint step {checkpoint.get('step')}", flush=True)
    image_loops = embed_images_student(student, processor, pictures, args.batch_size, device, args.num_loops)
    text_loops = embed_texts_student(student, processor, texts, args.batch_size, args.max_length, device, args.num_loops)
    for index, (image_s, text_s) in enumerate(zip(image_loops, text_loops)):
        image_to_text, text_to_image_score = recall_retrieval(image_s, text_s, text_to_image, image_to_texts)
        print(f"midtrain-loop{index:<4} I2T {image_to_text} T2I {text_to_image_score}", flush=True)


if __name__ == "__main__":
    main()
