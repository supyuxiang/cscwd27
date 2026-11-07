"""Convert raw Weibo or Twitter files into JSONL manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rawdata import load_twitter_table, load_weibo_splits, write_jsonl


def prepare_weibo(root: Path, out: Path) -> None:
    splits = load_weibo_splits(root)
    write_jsonl(out / "train.jsonl", splits["train"])
    write_jsonl(out / "test.jsonl", splits["test"])
    print(f"train: {len(splits['train'])} posts, skipped missing images: {splits['train_missing_images']}")
    print(f"test:  {len(splits['test'])} posts, skipped missing images: {splits['test_missing_images']}")
    print(f"wrote {out / 'train.jsonl'}")
    print(f"wrote {out / 'test.jsonl'}")
    print("Set image_root in the yaml config to:")
    print(f"  {root.resolve()}")


def prepare_twitter(root: Path, out: Path, table: Path | None, humor_as_fake: bool) -> None:
    if table is None:
        candidates = [
            root / "train.txt",
            root / "test.txt",
            root / "tweets.txt",
            root / "posts.txt",
        ]
        tables = [path for path in candidates if path.is_file()]
        if not tables:
            tables = sorted(root.glob("*.tsv")) + sorted(root.glob("*.csv"))
        if not tables:
            raise FileNotFoundError(
                "Pass --table, or place train/test txt/csv/tsv files in the Twitter root."
            )
    else:
        tables = [table]
    for path in tables:
        samples = load_twitter_table(path, root, humor_as_fake=humor_as_fake)
        destination = out / f"{path.stem}.jsonl"
        write_jsonl(destination, samples)
        print(f"{path.name}: {len(samples)} posts -> {destination}")
    print("Set image_root in the yaml config to:")
    print(f"  {root.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Weibo or Twitter manifests for CMAAN.")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    weibo = subparsers.add_parser("weibo", help="Jin et al. ACM MM 2017 Weibo rumor set")
    weibo.add_argument("--root", type=Path, required=True, help="Directory with txt files and images")
    weibo.add_argument("--out", type=Path, default=Path("data/weibo"))

    twitter = subparsers.add_parser("twitter", help="MediaEval-style Twitter rumor set")
    twitter.add_argument("--root", type=Path, required=True, help="Directory that contains the images")
    twitter.add_argument("--table", type=Path, default=None, help="CSV/TSV/TXT table. Default: search --root")
    twitter.add_argument("--out", type=Path, default=Path("data/twitter"))
    twitter.add_argument(
        "--humor-as-fake",
        action="store_true",
        help="Map humor labels to fake. Default is to drop humor posts.",
    )

    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.dataset == "weibo":
        prepare_weibo(args.root, args.out)
    else:
        prepare_twitter(args.root, args.out, args.table, args.humor_as_fake)


if __name__ == "__main__":
    main()
