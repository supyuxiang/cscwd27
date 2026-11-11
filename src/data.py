from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


def _resolve_image(path_str: str, root: Path | None) -> Path | None:
    path = Path(path_str)
    if path.is_file():
        return path
    if root is None:
        return None
    for candidate in (root / path, root / path.name):
        if candidate.is_file():
            return candidate
    return None


def load_jsonl_samples(path: str | Path, image_root: str | Path | None = None) -> tuple[list[dict], int]:
    root = Path(image_root) if image_root else None
    samples: list[dict] = []
    missing = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            image = _resolve_image(str(obj["image"]), root)
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
    groups: dict[int, list[dict]] = {}
    for sample in samples:
        groups.setdefault(int(sample["label"]), []).append(sample)
    rng = random.Random(seed)
    train, val = [], []
    for group in groups.values():
        rng.shuffle(group)
        n_val = min(int(round(len(group) * val_ratio)), len(group) - 1)
        n_val = max(n_val, 0)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class LoopFuseDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        last_error: Exception | None = None
        for shift in range(len(self.samples)):
            sample = self.samples[
                (index + shift) % len(self.samples)
            ]
            try:
                image = Image.open(sample['image']).convert('RGB')
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


def loopfuse_collator(processor, max_length: int):

    def collate_fn(batch: list[dict]) -> dict:
        texts = [(sample["text"] or "").strip() or "." for sample in batch]
        encoded = processor(
            text=texts,
            images=[sample["image"] for sample in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded["labels"] = torch.tensor([int(sample["label"]) for sample in batch], dtype=torch.long)
        return encoded

    return collate_fn


def test():
    """Tiny offline demo. Run: python -m src.data"""
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="loopfuse-data-"))
    for index, _label in enumerate([0, 1, 0, 1]):
        Image.new("RGB", (16, 16), (index * 40, 0, 0)).save(root / f"{index}.png")
    (root / "broken.png").write_bytes(b"not an image")
    rows = [
        {"id": "0", "text": "", "image": "0.png", "label": 0},
        {"id": "1", "text": "a real post", "image": "1.png", "label": 1},
        {"id": "2", "text": "another post", "image": "2.png", "label": 0},
        {"id": "3", "text": "a fake post", "image": "3.png", "label": 1},
        {"id": "missing", "text": "gone", "image": "nope.png", "label": 1},
        {"id": "broken", "text": "bad file", "image": "broken.png", "label": 0},
    ]
    manifest = root / "train.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    samples, missing = load_jsonl_samples(manifest, root)
    train, val = stratified_split(samples, val_ratio=0.5, seed=0)
    dataset = LoopFuseDataset(samples)
    item = dataset[0]
    skipped = dataset[next(i for i, sample in enumerate(samples) if sample["id"] == "broken")]

    class Processor:
        def __call__(self, text, images, return_tensors, padding, truncation, max_length):
            print("collate texts:", text)
            print("collate image sizes:", [image.size for image in images])
            width = min(4, max_length)
            batch = len(text)
            return {
                "input_ids": torch.ones(batch, width, dtype=torch.long),
                "attention_mask": torch.ones(batch, width, dtype=torch.long),
                "pixel_values": torch.zeros(batch, 3, 16, 16),
            }

    batch = loopfuse_collator(Processor(), max_length=8)([dataset[0], dataset[1]])
    print(f"loaded={len(samples)} missing_images={missing} train={len(train)} val={len(val)}")
    print(f"item id={item['id']} label={item['label']} image={item['image'].size} text={item['text']!r}")
    print(f"broken image skipped -> id={skipped['id']}")
    for key, value in batch.items():
        print(f"batch[{key}] {tuple(value.shape)} {value.dtype}")
    print("demo dir", root)


if __name__ == '__main__':
    test()
    