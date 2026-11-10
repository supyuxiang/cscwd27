from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


def resolve_image(path_str: str, image_root: str | Path | None) -> Path | None:
    path = Path(path_str)
    if path.is_file():
        return path
    if image_root is None or str(image_root).strip() == "":
        return None
    root = Path(image_root)
    for candidate in (root / path, root / path.name):
        if candidate.is_file():
            return candidate
    return None


def load_jsonl_samples(path: str | Path, image_root: str | Path | None = None) -> tuple[list[dict], int]:
    samples = []
    missing = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            image = resolve_image(str(obj["image"]), image_root)
            if image is None:
                missing += 1
                continue
            samples.append(
                {
                    "id": str(obj.get("id", len(samples))),
                    "text": str(obj.get("text") or ""),
                    "image": str(image),
                    "label": int(obj["label"]),
                }
            )
    if not samples:
        raise RuntimeError(f"No usable samples in {path}. Missing images: {missing}")
    return samples, missing


def stratified_split(samples: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if val_ratio <= 0:
        return list(samples), []
    by_label: dict[int, list[dict]] = {}
    for sample in samples:
        by_label.setdefault(int(sample["label"]), []).append(sample)
    rng = random.Random(seed)
    train_samples: list[dict] = []
    val_samples: list[dict] = []
    for group in by_label.values():
        items = list(group)
        rng.shuffle(items)
        n_val = int(round(len(items) * val_ratio))
        if len(items) - n_val < 1:
            n_val = 0
        val_samples.extend(items[:n_val])
        train_samples.extend(items[n_val:])
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


class MultimodalDataset(Dataset):
    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        count = len(self.samples)
        last_error: Exception | None = None
        for shift in range(count):
            sample = self.samples[(index + shift) % count]
            try:
                image = Image.open(sample["image"]).convert("RGB")
            except OSError as exc:
                last_error = exc
                continue
            return {
                "id": sample["id"],
                "text": sample["text"],
                "image": image,
                "label": int(sample["label"]),
            }
        raise RuntimeError(f"Failed to open images starting at index {index}: {last_error}")


def build_collate(processor, max_text_length: int):
    def collate(batch: list[dict]) -> dict:
        texts = []
        for sample in batch:
            text = (sample["text"] or "").strip()
            texts.append(text if text else ".")
        encoded = processor(
            text=texts,
            images=[sample["image"] for sample in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_text_length,
        )
        encoded["labels"] = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
        return encoded

    return collate





class LoopFuseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
    ):
        super().__init__()
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass



def loopfuse_collator(batch):
    return


