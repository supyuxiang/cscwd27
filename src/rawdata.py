"""Parsers for the public Weibo and Twitter multimodal rumor datasets."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
FAKE_LABELS = {"1", "fake", "rumor", "rumour", "false", "yes"}
REAL_LABELS = {"0", "real", "nonrumor", "non-rumor", "non_rumor", "true", "no"}
HUMOR_LABELS = {"humor", "humour"}
TEXT_COLUMNS = ("text", "post_text", "tweet", "tweet_text", "content", "post", "tweettext")
IMAGE_COLUMNS = ("image", "image_path", "image_id", "img", "image_url", "imageid", "images")
LABEL_COLUMNS = ("label", "target", "y")
ID_COLUMNS = ("id", "post_id", "tweet_id", "mid", "tweetid")


def read_text(path: Path) -> str:
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        "unknown", b"", 0, 1, f"Cannot decode {path}: {last_error}"
    )


def looks_like_image_line(line: str) -> bool:
    text = line.strip().lower()
    return "http" in text or "null" in text or "|" in text


def parse_weibo_txt(path: Path) -> list[dict]:
    """Parse Jin et al. Weibo files: metadata, image urls, text, repeated."""
    raw_lines = read_text(path).splitlines()
    while raw_lines and raw_lines[-1].strip() == "":
        raw_lines.pop()
    groups = _best_weibo_groups(raw_lines)
    records = []
    for group in groups:
        if len(group) < 3 or not looks_like_image_line(group[1]):
            continue
        tweet_id = group[0].split("|", 1)[0].strip()
        urls = [
            piece.strip()
            for piece in group[1].split("|")
            if piece.strip() and piece.strip().lower() != "null"
        ]
        records.append({"id": tweet_id, "urls": urls, "text": group[2].strip()})
    if not records:
        raise RuntimeError(f"No Weibo posts parsed from {path}")
    return records


def _best_weibo_groups(lines: list[str]) -> list[list[str]]:
    candidates: list[list[list[str]]] = []
    if len(lines) >= 3:
        candidates.append([lines[i : i + 3] for i in range(0, len(lines) - 2, 3)])
    nonempty = [line for line in lines if line.strip() != ""]
    if len(nonempty) >= 3:
        candidates.append([nonempty[i : i + 3] for i in range(0, len(nonempty) - 2, 3)])

    blocks: list[list[str]] = []
    buffer: list[str] = []
    for line in lines:
        if line.strip() == "":
            if buffer:
                blocks.append(buffer)
                buffer = []
        else:
            buffer.append(line)
    if buffer:
        blocks.append(buffer)
    if blocks and all(len(block) >= 3 for block in blocks):
        candidates.append([block[:3] for block in blocks])

    def score(groups: list[list[str]]) -> float:
        if not groups:
            return -1.0
        hits = sum(1 for group in groups if len(group) >= 2 and looks_like_image_line(group[1]))
        return hits / len(groups)

    best = max(candidates, key=score)
    if score(best) < 0.5:
        raise RuntimeError("Could not recognize the Weibo three-line post format.")
    return best


def index_images(root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    exact: dict[str, Path] = {}
    stems: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        exact[path.name.lower()] = path
        stems.setdefault(path.stem.lower(), path)
    return exact, stems


def find_named(root: Path, name: str) -> Path | None:
    matches = [path for path in root.rglob(name) if path.is_file()]
    if not matches:
        return None
    matches.sort(key=lambda path: (len(path.parts), str(path)))
    return matches[0]


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def match_image(urls: list[str], exact: dict[str, Path], stems: dict[str, Path]) -> Path | None:
    for url in urls:
        name = url.split("?")[0].rstrip("/").split("/")[-1].lower()
        if not name:
            continue
        if name in exact:
            return exact[name]
        stem = Path(name).stem.lower()
        if stem in stems:
            return stems[stem]
    return None


def load_weibo_splits(root: Path) -> dict[str, list[dict]]:
    specs = {
        "train": (("train_rumor.txt", 1), ("train_nonrumor.txt", 0)),
        "test": (("test_rumor.txt", 1), ("test_nonrumor.txt", 0)),
    }
    exact, stems = index_images(root)
    if not exact:
        raise RuntimeError(f"No images found under {root}")
    splits: dict[str, list[dict]] = {}
    for split, files in specs.items():
        samples = []
        missing = 0
        for filename, label in files:
            path = find_named(root, filename)
            if path is None:
                raise FileNotFoundError(f"Missing {filename} under {root}")
            for record in parse_weibo_txt(path):
                image = match_image(record["urls"], exact, stems)
                if image is None:
                    missing += 1
                    continue
                samples.append(
                    {
                        "id": record["id"],
                        "text": record["text"],
                        "image": _relative_to_root(image, root),
                        "label": label,
                    }
                )
        if not samples:
            raise RuntimeError(f"No usable {split} samples. Missing images: {missing}")
        splits[split] = samples
        splits[f"{split}_missing_images"] = missing  # type: ignore[assignment]
    return splits


def parse_label(value: str, humor_as_fake: bool = False) -> int | None:
    text = str(value).strip().lower()
    if text in FAKE_LABELS:
        return 1
    if text in REAL_LABELS:
        return 0
    if text in HUMOR_LABELS:
        return 1 if humor_as_fake else None
    raise ValueError(f"Unrecognized label: {value!r}")


def _pick(row: dict, names: tuple[str, ...]) -> str | None:
    lowered = {str(key).strip().lower(): value for key, value in row.items() if key is not None}
    for name in names:
        value = lowered.get(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def _split_image_field(field: str) -> list[str]:
    return [part for part in re.split(r"[|,;\s]+", field.strip()) if part and part.lower() != "null"]


def load_twitter_table(path: Path, image_root: Path, humor_as_fake: bool = False) -> list[dict]:
    rows = _read_rows(path)
    exact, stems = index_images(image_root)
    samples = []
    skipped = 0
    for row in rows:
        label_text = _pick(row, LABEL_COLUMNS)
        text = _pick(row, TEXT_COLUMNS)
        image_field = _pick(row, IMAGE_COLUMNS)
        if label_text is None or text is None or image_field is None:
            skipped += 1
            continue
        label = parse_label(label_text, humor_as_fake=humor_as_fake)
        if label is None:
            skipped += 1
            continue
        image = match_image(_split_image_field(image_field), exact, stems)
        if image is None:
            direct = Path(image_field)
            if not direct.is_file():
                direct = image_root / image_field
            image = direct if direct.is_file() else None
        if image is None:
            skipped += 1
            continue
        samples.append(
            {
                "id": _pick(row, ID_COLUMNS) or str(len(samples)),
                "text": text,
                "image": _relative_to_root(image, image_root),
                "label": label,
            }
        )
    if not samples:
        raise RuntimeError(f"No usable Twitter samples in {path}. Skipped rows: {skipped}")
    return samples


def _read_rows(path: Path) -> list[dict]:
    sample = path.read_text(encoding="utf-8", errors="replace")[:8192]
    delimiter = "\t" if sample.count("\t") >= sample.count(",") else ","
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        raise RuntimeError(f"Empty table: {path}")
    header = [cell.strip().lower() for cell in rows[0]]
    has_header = any(name in header for name in ("label", "text", "tweet_text", "tweettext", "image_id"))
    if has_header:
        dict_rows = []
        for cells in rows[1:]:
            padded = cells + [""] * (len(header) - len(cells))
            dict_rows.append({header[i]: padded[i] for i in range(len(header))})
        return dict_rows
    # MediaEval-style rows without a header:
    # tweetId, tweetText, userId, imageId, username, timestamp, label
    dict_rows = []
    for cells in rows:
        if len(cells) < 7:
            continue
        dict_rows.append(
            {
                "tweet_id": cells[0],
                "text": cells[1],
                "image_id": cells[3],
                "label": cells[6],
            }
        )
    return dict_rows


def write_jsonl(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
