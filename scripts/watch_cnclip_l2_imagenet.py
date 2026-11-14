"""Submit Flickr evals for the ImageNet loop-2 mid-train and mirror metrics to SwanLab.

Training writes ``metrics.jsonl``. Each eval writes ``eval/step_<label>.json``.
This process is the only SwanLab client, so a finished eval cannot close the run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/yxfeng/cscwd27_loopfuse")
SBATCH = ROOT / "scripts/eval_cnclip_l2_imagenet.sbatch"
OUT = Path(os.environ.get("CNCLIP_MID_OUT", "/data3/yxfeng/experiments/cnclip_midtrain_l2_imagenet_zh"))
KEY_FILE = ROOT / "logs/swanlab_api_key"
MIN_CKPT_BYTES = 2_000_000_000


def job_state(job_id: str) -> str:
    result = subprocess.run(
        ["squeue", "-j", job_id, "-h", "-o", "%T"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def sacct_state(job_id: str) -> str:
    result = subprocess.run(
        ["sacct", "-j", job_id, "-n", "-X", "-o", "State"],
        check=False,
        capture_output=True,
        text=True,
    )
    line = result.stdout.strip().splitlines()
    return line[0].split()[0] if line else ""


def stable(path: Path) -> bool:
    if not path.is_file():
        return False
    size_a = path.stat().st_size
    time.sleep(8)
    size_b = path.stat().st_size
    return size_a == size_b and size_a > MIN_CKPT_BYTES


def read_new_lines(path: Path, offset: int) -> tuple[int, list[dict]]:
    if not path.is_file():
        return 0, []
    size = path.stat().st_size
    if size < offset:
        offset = 0
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
    if not data:
        return offset, []
    if not data.endswith(b"\n"):
        cut = data.rfind(b"\n")
        if cut < 0:
            return offset, []
        data = data[: cut + 1]
    rows = []
    for line in data.decode().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return offset + len(data), rows


def load_jobs(path: Path) -> dict[str, list[str]]:
    jobs: dict[str, list[str]] = {}
    if not path.is_file():
        return jobs
    for line in path.read_text().splitlines():
        label, job_id = line.split()
        jobs.setdefault(label, []).append(job_id)
    return jobs


def submit_eval(label: str, checkpoint: Path, jobs_path: Path) -> str:
    result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--export=NONE,CKPT={checkpoint},LABEL={label},CNCLIP_MID_OUT={OUT}",
            str(SBATCH),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = result.stdout.strip()
    with jobs_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{label} {job_id}\n")
    print(f"EVAL_SUBMIT step {label} job {job_id}", flush=True)
    return job_id


def train_saved(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    for line in log_path.read_text(errors="replace").splitlines():
        if line.startswith("saved "):
            return True
    return False


def api_key() -> str:
    key = os.environ.get("SWANLAB_API_KEY", "").strip()
    if key:
        return key
    if KEY_FILE.is_file():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    return ""


def open_swanlab():
    import swanlab

    key = api_key()
    if not key:
        return None
    os.environ["SWANLAB_API_KEY"] = key
    swanlab.login(api_key=key, save=False)
    kwargs = {
        "project": os.environ.get("SWANLAB_PROJECT", "cnclip-midtrain"),
        "name": os.environ.get("SWANLAB_EXP", "loop2-imagenet-zh"),
        "mode": os.environ.get("SWANLAB_MODE", "online"),
        "log_dir": str(OUT / "swanlog"),
        "config": {
            "num_loops": 2,
            "loss": "last",
            "data": "imagenet-train-zh",
            "init": os.environ.get(
                "CNCLIP_MID_INIT",
                "/data3/yxfeng/experiments/cnclip_midtrain_l2_loop1loss/latest.pt",
            ),
            "gpus": 8,
            "node": "node2",
            "batch_per_gpu": 32,
            "global_batch": 256,
            "steps": 10010,
            "lr": 5e-6,
            "weight_decay": 0.01,
            "lr_warmup": 400,
            "loop_warmup": 0,
            "save_every": 2000,
        },
    }
    workspace = os.environ.get("SWANLAB_WORKSPACE", "").strip()
    if workspace:
        kwargs["workspace"] = workspace
    run = swanlab.init(**kwargs)
    (OUT / "swanlab_id.txt").write_text(run.id + "\n", encoding="utf-8")
    try:
        print(f"swanlab {run.url}", flush=True)
        (OUT / "swanlab_url.txt").write_text(run.url + "\n", encoding="utf-8")
    except Exception:
        print(f"swanlab id {run.id}", flush=True)
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-job", required=True)
    parser.add_argument("--train-log", required=True)
    args = parser.parse_args()
    train_log = Path(args.train_log)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    jobs_path = OUT / "eval_jobs.txt"
    run = open_swanlab()
    announced_missing_key = run is not None

    offset_path = OUT / "metrics.offset"
    seen_path = OUT / "swanlab_eval_steps.txt"
    offset = int(offset_path.read_text().strip()) if offset_path.is_file() and offset_path.read_text().strip() else 0
    seen_eval_steps = set()
    if seen_path.is_file():
        seen_eval_steps = {int(line) for line in seen_path.read_text().splitlines() if line.strip()}
    grace: dict[str, int] = {}
    failed: set[str] = set()
    print(f"WATCH_START train={args.train_job} out={OUT}", flush=True)

    while True:
        if run is None:
            run = open_swanlab()
            if run is None and not announced_missing_key:
                print("swanlab key missing; Flickr evals still run. Write the key to logs/swanlab_api_key", flush=True)
                announced_missing_key = True
        if run is not None:
            offset, rows = read_new_lines(OUT / "metrics.jsonl", offset)
            for row in rows:
                run.log(row["metrics"], step=int(row["step"]))
            if rows:
                offset_path.write_text(f"{offset}\n", encoding="utf-8")
                print(f"swanlab train lines {len(rows)} through step {rows[-1]['step']}", flush=True)

            for path in sorted((OUT / "eval").glob("step_*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                step = int(payload["step"])
                if step in seen_eval_steps:
                    continue
                run.log(payload["metrics"], step=step)
                seen_eval_steps.add(step)
                seen_path.write_text("".join(f"{item}\n" for item in sorted(seen_eval_steps)), encoding="utf-8")
                print(f"swanlab eval step {step} from {path.name}", flush=True)

        jobs = load_jobs(jobs_path)
        checkpoints = sorted(OUT.glob("step_*.pt"))
        labels = [path.name[len("step_") : -len(".pt")] for path in checkpoints]
        if not job_state(args.train_job) and train_saved(train_log):
            labels.append("final")

        for label in labels:
            metrics_path = OUT / "eval" / f"step_{label}.json"
            if metrics_path.is_file():
                continue
            history = jobs.get(label, [])
            if history and job_state(history[-1]):
                continue
            if history and not job_state(history[-1]):
                grace[label] = grace.get(label, 0) + 1
                if grace[label] < 2:
                    continue
            if len(history) >= 2:
                if label not in failed:
                    print(f"EVAL_FAIL step {label} after {len(history)} jobs", flush=True)
                    failed.add(label)
                continue
            checkpoint = OUT / "latest.pt" if label == "final" else OUT / f"step_{label}.pt"
            if not stable(checkpoint):
                print(f"writing step {label}", flush=True)
                continue
            grace[label] = 0
            submit_eval(label, checkpoint, jobs_path)

        train_alive = bool(job_state(args.train_job))
        if not train_alive:
            time.sleep(20)
            pending = False
            for label in labels:
                metrics_path = OUT / "eval" / f"step_{label}.json"
                history = load_jobs(jobs_path).get(label, [])
                if metrics_path.is_file():
                    continue
                if history and job_state(history[-1]):
                    pending = True
                    continue
                if len(history) < 2:
                    pending = True
            if train_saved(train_log) and not (OUT / "eval" / "step_final.json").is_file():
                final_jobs = load_jobs(jobs_path).get("final", [])
                if len(final_jobs) < 2 or (final_jobs and job_state(final_jobs[-1])):
                    pending = True
            if not pending:
                if run is not None:
                    offset, rows = read_new_lines(OUT / "metrics.jsonl", offset)
                    for row in rows:
                        run.log(row["metrics"], step=int(row["step"]))
                    if rows:
                        offset_path.write_text(f"{offset}\n", encoding="utf-8")
                    for path in sorted((OUT / "eval").glob("step_*.json")):
                        payload = json.loads(path.read_text(encoding="utf-8"))
                        step = int(payload["step"])
                        if step in seen_eval_steps:
                            continue
                        run.log(payload["metrics"], step=step)
                        seen_eval_steps.add(step)
                    run.finish()
                state = sacct_state(args.train_job)
                print(f"WATCH_DONE train={state}", flush=True)
                return
        time.sleep(30)


if __name__ == "__main__":
    main()
