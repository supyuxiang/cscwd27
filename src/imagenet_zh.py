"""ImageNet-1k images paired with Chinese-CLIP class prompts.

Class order is the official 1,000-synset order. ``label_cn.txt`` is the
Chinese-CLIP translation of that order, so folder ``n01440764`` is 丁鲷.
Each sample draws one photo-style template. The zero-shot list also contains
sculpture, tattoo, and cartoon prompts; those are not used here, because the
ImageNet files are photographs.
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

ZH_PHOTO_TEMPLATES = (
    "{}的照片。",
    "一张{}的照片。",
    "一张干净的{}的照片。",
    "一张包含{}的照片。",
    "我的{}的照片。",
    "{}的特写照片。",
    "{}的相片。",
    "一张{}的好照片。",
    "高质量的{}的照片。",
    "一张关于{}的照片。",
    "{}的明亮照片。",
    "一张{}的明亮照片。",
)


def load_zh_labels(train_root: Path, label_file: Path) -> dict[str, str]:
    synsets = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    labels = [line.strip() for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(synsets) != 1000 or len(labels) != 1000:
        raise RuntimeError(f"expected 1000 classes, got folders={len(synsets)} labels={len(labels)}")
    if synsets[0] != "n01440764" or labels[0] != "丁鲷":
        raise RuntimeError(f"label order mismatch: {synsets[0]} {labels[0]}")
    return dict(zip(synsets, labels))


class ImageNetZhStream(IterableDataset):
    """Cycle the ImageNet train split. Each rank and worker reads a disjoint set."""

    def __init__(self, train_root: Path, label_file: Path, rank: int, world: int, seed: int):
        super().__init__()
        self.rank = rank
        self.world = world
        self.seed = seed
        labels = load_zh_labels(train_root, label_file)
        samples: list[tuple[str, str]] = []
        for folder in sorted(path for path in train_root.iterdir() if path.is_dir()):
            label = labels[folder.name]
            for image in folder.iterdir():
                if image.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                    samples.append((str(image), label))
        if not samples:
            raise RuntimeError(f"no images under {train_root}")
        self.samples = samples

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        stride = self.world * workers
        offset = self.rank * workers + worker_id
        rng = random.Random(self.seed + offset)
        chosen = list(range(offset, len(self.samples), stride))
        while True:
            rng.shuffle(chosen)
            for index in chosen:
                path, label = self.samples[index]
                text = rng.choice(ZH_PHOTO_TEMPLATES).format(label)
                try:
                    with Image.open(path) as picture:
                        picture.load()
                        picture = picture.convert("RGB")
                except (OSError, SyntaxError, ValueError):
                    continue
                yield {"text": text, "image": picture}
