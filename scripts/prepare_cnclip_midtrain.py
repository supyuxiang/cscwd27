"""Download longer Chinese CLIP pretraining pairs.

Wukong captions are at most 50 characters. LAION has about 7.0 million
captions of 51 characters or more. Those are written longest-first and
downloaded until the requested number of images succeeds. If that tail cannot
supply the target, captions of 40 to 50 characters are used only to fill the
remainder. Saved images are JPEG with the long side at most 384 pixels.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path("/data3/yxfeng/datasets/cnclip_pretrain")
USER_AGENT = "Mozilla/5.0 (compatible; cnclip-midtrain/1.0)"


def clean_caption(text: str | None, low: int, high: int | None) -> str | None:
    if not text:
        return None
    cleaned = " ".join(str(text).replace("\x00", " ").split())
    if len(cleaned) < low:
        return None
    if high is not None and len(cleaned) > high:
        return None
    return cleaned


def collect_laion(files: list[Path], low: int, high: int | None, max_rows: int, limit: int) -> list[tuple[int, str, str]]:
    kept: list[tuple[int, str, str]] = []
    scanned = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        columns = ["URL", "TEXT", "similarity", "NSFW"]
        for batch in parquet.iter_batches(batch_size=65536, columns=columns):
            for row in batch.to_pylist():
                scanned += 1
                if max_rows and scanned > max_rows:
                    kept.sort(key=lambda item: item[0], reverse=True)
                    print(f"scan stopped at {max_rows}, kept {len(kept)} chars {low}-{high}", flush=True)
                    return kept
                if row.get("NSFW") == "NSFW":
                    continue
                if row.get("similarity") is None or row["similarity"] < 0.26:
                    continue
                text = clean_caption(row.get("TEXT"), low, high)
                url = row.get("URL")
                if text is None or not isinstance(url, str) or not url.startswith("http"):
                    continue
                kept.append((len(text), url, text))
                if limit and len(kept) >= limit:
                    kept.sort(key=lambda item: item[0], reverse=True)
                    print(f"scan kept {len(kept)} chars {low}-{high} after {scanned} rows", flush=True)
                    return kept
            if scanned % 5_000_000 < 65536:
                print(f"scanned {scanned} kept {len(kept)} chars {low}-{high}", flush=True)
    kept.sort(key=lambda item: item[0], reverse=True)
    print(f"scan done scanned {scanned} kept {len(kept)} chars {low}-{high}", flush=True)
    return kept


def write_candidates(path: Path, rows: list[tuple[int, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for length, url, text in rows:
            handle.write(json.dumps({"chars": length, "url": url, "text": text}, ensure_ascii=False) + "\n")


def fetch_image(url: str, timeout: float) -> bytes | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read(2_000_001)
    except Exception:
        return None
    if len(payload) < 1024 or len(payload) > 2_000_000:
        return None
    if payload[:3] == b"\xff\xd8\xff" or payload[:8] == b"\x89PNG\r\n\x1a\n" or payload[:4] == b"RIFF":
        return payload
    return None


def image_path(root: Path, index: int) -> Path:
    return root / "images" / "laion" / f"{index // 1000:06d}" / f"{index:09d}.jpg"


def load_done(jsonl_dir: Path) -> set[int]:
    done: set[int] = set()
    if not jsonl_dir.is_dir():
        return done
    for path in jsonl_dir.glob("laion-*.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done.add(int(json.loads(line)["id"]))
    return done


def download(root: Path, candidates: list[Path], target: int, workers: int, timeout: float) -> int:
    jsonl_dir = root / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    done = load_done(jsonl_dir)
    success = len(done)
    print(f"already saved {success}, target {target}", flush=True)
    if success >= target:
        return success

    pending: queue.Queue = queue.Queue(maxsize=workers * 4)
    stop = threading.Event()
    lock = threading.Lock()
    started = time.time()
    attempts = 0

    def reader() -> None:
        index = 0
        for path in candidates:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if stop.is_set():
                        break
                    if not line.strip():
                        continue
                    current = index
                    index += 1
                    if current in done:
                        continue
                    while not stop.is_set():
                        try:
                            pending.put((current, line), timeout=0.5)
                            break
                        except queue.Full:
                            continue
            if stop.is_set():
                break
        for _ in range(workers):
            while True:
                try:
                    pending.put(None, timeout=0.5)
                    break
                except queue.Full:
                    if stop.is_set() and pending.qsize() > workers:
                        continue

    def consume(worker_id: int) -> None:
        nonlocal success, attempts
        out_path = jsonl_dir / f"laion-{worker_id:03d}.jsonl"
        with out_path.open("a", encoding="utf-8") as handle:
            while True:
                item = pending.get()
                if item is None:
                    return
                index, line = item
                if stop.is_set():
                    continue
                row = json.loads(line)
                destination = image_path(root, index)
                exists = destination.is_file() and destination.stat().st_size > 500
                if not exists:
                    payload = fetch_image(row["url"], timeout)
                    with lock:
                        attempts += 1
                        tried = attempts
                    if tried % 2000 == 0:
                        elapsed = max(time.time() - started, 1e-6)
                        print(f"tried {tried} saved {success} {success / elapsed:.1f} img/s", flush=True)
                    if payload is None:
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(".part")
                    temporary.write_bytes(payload)
                    temporary.replace(destination)
                record = {"id": index, "chars": row.get("chars"), "text": row["text"], "image": str(destination)}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                with lock:
                    success += 1
                    current = success
                    if current >= target:
                        stop.set()
                    if current % 1000 == 0:
                        rate = current / max(time.time() - started, 1e-6)
                        print(f"saved {current}/{target} {rate:.1f} img/s", flush=True)

    threads = [threading.Thread(target=consume, args=(i,), daemon=True) for i in range(workers)]
    producer = threading.Thread(target=reader, daemon=True)
    producer.start()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stop.set()
    producer.join(timeout=5)
    print(f"finished saved {success}", flush=True)
    return success


def candidate_file(root: Path, low: int, high: int | None) -> Path:
    upper = "inf" if high is None else str(high)
    return root / "candidates" / f"laion_{low}_{upper}.jsonl"


def ensure_band(root: Path, files: list[Path], low: int, high: int | None, max_rows: int, limit: int = 0) -> Path:
    path = candidate_file(root, low, high)
    meta = path.with_suffix(".meta.json")
    if path.is_file() and meta.is_file() and not max_rows and not limit:
        print(f"reuse {path}", flush=True)
        return path
    rows = collect_laion(files, low, high, max_rows, limit)
    write_candidates(path, rows)
    meta.write_text(json.dumps({"count": len(rows), "min_chars": low, "max_chars": high}) + "\n")
    print(f"wrote {path} lines {len(rows)}", flush=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download longer LAION Chinese image-text pairs.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tables-root", type=Path, default=None)
    parser.add_argument("--target", type=int, default=5_000_000)
    parser.add_argument("--min-chars", type=int, default=51)
    parser.add_argument("--fill-chars", type=int, default=40, help="Shorter floor used only if the long tail is not enough.")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    tables = args.tables_root or root
    files = sorted((tables / "laion2b_zh").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No LAION parquet files under {tables}")
    long_path = ensure_band(root, files, args.min_chars, None, args.max_rows)
    saved = download(root, [long_path], args.target, args.workers, args.timeout)
    used_fill = False
    if saved < args.target and args.fill_chars < args.min_chars:
        print(f"long tail saved {saved}, filling from {args.fill_chars}-{args.min_chars - 1} characters", flush=True)
        fill_path = ensure_band(root, files, args.fill_chars, args.min_chars - 1, args.max_rows, limit=1_000_000)
        saved = download(root, [long_path, fill_path], args.target, args.workers, args.timeout)
        used_fill = True
    summary = {"laion": saved, "wukong": 0, "total": saved, "target": args.target, "min_chars": args.min_chars, "used_fill": used_fill}
    (root / "counts.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if saved < args.target:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
