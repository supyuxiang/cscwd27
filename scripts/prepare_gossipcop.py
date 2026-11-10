"""Export the public image-text GossipCop release into LoopFuse jsonl.

Source: https://huggingface.co/datasets/Ahren09/MMSoc_GossipCop
This is the MM-Soc image-text pack of FakeNewsNet GossipCop. The BMR /
SpotFake+ split cited by later papers (train 10010 = 7974 real / 2036 fake,
test 2830 = 2285 real / 545 fake) is distributed by email, not as a public
archive. This release is the same collection with broken images removed:
train 9988 (7955 real / 2033 fake), test 2672 (2169 real / 503 fake).

Labels match Weibo: 0 real, 1 fake.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

REPO = "Ahren09/MMSoc_GossipCop"
IMAGE_ROOT = Path("/home/yxfeng/datasets/cscwd27/gossipcop/images")
JSONL_ROOT = Path("/home/yxfeng/cscwd27_loopfuse/data/gossipcop")


def export_split(dataset, split: str) -> dict:
    out_dir = IMAGE_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = JSONL_ROOT / f"{split}.jsonl"
    counts = {"real": 0, "fake": 0, "skipped": 0, "empty_text": 0}
    written = 0
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(dataset):
            label = int(sample["label"])
            if label not in (0, 1):
                counts["skipped"] += 1
                continue
            image = sample["image"]
            if image is None:
                counts["skipped"] += 1
                continue
            sample_id = f"gossipcop-{split}-{index:05d}"
            image_path = out_dir / f"{sample_id}.jpg"
            if not image_path.is_file():
                try:
                    image.convert("RGB").save(image_path, format="JPEG", quality=95)
                except Exception:
                    counts["skipped"] += 1
                    image_path.unlink(missing_ok=True)
                    continue
            text = str(sample.get("text") or "").strip()
            if not text:
                text = "."
                counts["empty_text"] += 1
            record = {
                "id": sample_id,
                "text": text,
                "image": str(image_path),
                "label": label,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["real" if label == 0 else "fake"] += 1
            written += 1
            if written % 500 == 0:
                print(f"{split}: wrote {written}", flush=True)
    counts["written"] = written
    print(f"{split}: {counts}", flush=True)
    return counts


def main() -> None:
    from datasets import load_dataset

    JSONL_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    dataset = load_dataset(REPO)
    stats = {
        "source": REPO,
        "note": (
            "Public MM-Soc image-text GossipCop. "
            "BMR split 10010/2830 is not in this archive."
        ),
        "splits": {},
    }
    for split in ("train", "test"):
        stats["splits"][split] = export_split(dataset[split], split)
    stats_path = JSONL_ROOT / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"stats: {stats_path}", flush=True)


if __name__ == "__main__":
    main()
