from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.modeling_loopfuse import LoopFuse
from src.utils import build_dataloaders, move_batch, run_epoch_eval, set_seed


class SwanLogger:
    def __init__(self, enabled: bool, config: dict, output_dir: Path):
        self.enabled = enabled
        if not enabled:
            return
        import swanlab

        self._swanlab = swanlab
        swanlab.init(
            project=str(config.get("swanlab_project", "loopfuse")),
            name=str(config.get("run_name", "loopfuse")),
            config=config,
            mode=str(config.get("swanlab_mode", "local")),
            log_dir=str(output_dir / "swanlog"),
        )

    def log(self, data: dict, step: int | None = None) -> None:
        if self.enabled:
            self._swanlab.log(data, step=step)

    def finish(self) -> None:
        if self.enabled:
            self._swanlab.finish()
            self.enabled = False


class LoopFuseTrainer:
    def __init__(self, config: dict):
        self.config = config

        self._init_distributed()

        set_seed(int(config.get("seed", 42)))

        self.num_loops = int(config.get("num_loops", 3))
        self.epochs = int(config["epochs"])
        self.patience = int(config.get("patience", 0))
        self.accum_steps = max(int(config.get("grad_accum_steps", 1)), 1)
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        self.output_dir = Path(config.get("output_dir", "outputs/run"))
        if not self.output_dir.is_absolute():
            self.output_dir = ROOT / self.output_dir

        self.logger = self._build_swanlab_logger()

        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.train_sampler,
        ) = self._build_dataloader()

        self.model = self._build_models()

        self.optimizer = self._build_optimizer()

        self.scheduler = self._build_scheduler()

        self.loss_fn = self._build_loss()

        self.amp_dtype, self.scaler = self._amp()

    def _init_distributed(self) -> None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.distributed = world_size > 1
        if self.distributed:
            if not torch.cuda.is_available():
                raise RuntimeError("DDP training expects CUDA. Launch with torchrun.")
            dist.init_process_group(backend="nccl")
            self.local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.local_rank = 0
            self.rank = 0
            self.world_size = 1
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.is_main = (self.rank == 0)

    def _build_swanlab_logger(self) -> SwanLogger:
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        return SwanLogger(self.is_main, self.config, self.output_dir)

    def _build_dataloader(self):
        processor = AutoProcessor.from_pretrained(str(self.config["clip_name"]), trust_remote_code=True)
        return build_dataloaders(
            self.config,
            processor,
            distributed=self.distributed,
            rank=self.rank,
            world_size=self.world_size,
            is_main=self.is_main,
        )

    def _build_models(self) -> torch.nn.Module:
        num_heads_q = int(self.config.get("num_heads_q", self.config.get("n_heads", 8)))
        num_heads_kv = int(self.config.get("num_heads_kv", num_heads_q))
        model = LoopFuse(
            clip_path=str(self.config["clip_name"]),
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            dropout_p=float(self.config.get("dropout", 0.1)),
            num_classes=int(self.config.get("num_classes", 2)),
        )
        for parameter in model.parameters():
            parameter.requires_grad = True
        model.to(self.device)
        if self.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )
        if self.is_main:
            total = sum(parameter.numel() for parameter in model.parameters())
            print(f"full finetune: {total} parameters, loops={self.num_loops}, device={self.device}")
        return model

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=float(self.config["lr"]),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
        )

    def _build_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=float(self.config.get("min_lr", float(self.config["lr"]) * 0.01)),
        )

    def _build_loss(self) -> CrossEntropyLoss:
        return CrossEntropyLoss()

    def _amp(self):
        enabled = bool(self.config.get("amp", True)) and self.device.type == "cuda"
        if not enabled:
            return None, None
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, None
        try:
            scaler = torch.amp.GradScaler("cuda")
        except TypeError:
            scaler = torch.cuda.amp.GradScaler()
        return torch.float16, scaler

    def _unwrap(self) -> torch.nn.Module:
        return self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

    def _forward_loss(self, batch: dict) -> torch.Tensor:
        enabled = self.amp_dtype is not None
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype or torch.float32, enabled=enabled):
            logits = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                num_loops=self.num_loops,
            )
            return self.loss_fn(logits, batch["labels"])

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_count = 0
        for step, batch in enumerate(tqdm(self.train_loader, disable=not self.is_main, leave=False)):
            batch = move_batch(batch, self.device)
            sync = (step + 1) % self.accum_steps == 0 or (step + 1) == len(self.train_loader)
            sync_context = self.model.no_sync() if self.distributed and not sync else contextlib.nullcontext()
            with sync_context:
                loss = self._forward_loss(batch)
                backward_loss = loss / self.accum_steps
                if self.scaler is None:
                    backward_loss.backward()
                else:
                    self.scaler.scale(backward_loss).backward()
            total_loss += float(loss.detach().item()) * int(batch["labels"].size(0))
            total_count += int(batch["labels"].size(0))
            if not sync:
                continue
            if self.scaler is None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            else:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        if self.distributed:
            packed = torch.tensor([total_loss, total_count], dtype=torch.float64, device=self.device)
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            total_loss = float(packed[0].item())
            total_count = int(packed[1].item())
        return total_loss / max(total_count, 1)

    @torch.no_grad()
    def evaluate(self, loader) -> tuple[dict[str, float], str]:
        return run_epoch_eval(self._unwrap(), loader, self.device, self.num_loops)

    def _save(self, name: str, epoch: int, metrics: dict | None) -> None:
        if not self.is_main:
            return
        path = self.output_dir / name
        state = {key: value.detach().cpu() for key, value in self._unwrap().state_dict().items()}
        torch.save({"model": state, "epoch": epoch, "metrics": metrics, "config": self.config}, path)

    def _should_stop(self, stop: bool) -> bool:
        if not self.distributed:
            return stop
        flag = torch.tensor([1 if stop else 0], device=self.device)
        dist.broadcast(flag, src=0)
        return bool(flag.item())

    def train(self) -> None:
        best_score = -1.0
        stale = 0
        for epoch in range(1, self.epochs + 1):
            loss = self._train_one_epoch(epoch)
            self.scheduler.step()
            metrics = None
            stop = False
            if self.is_main:
                message = f"epoch {epoch}/{self.epochs} loss {loss:.4f} lr {self.scheduler.get_last_lr()[0]:.2e}"
                logged = {"train/loss": loss, "lr": self.scheduler.get_last_lr()[0]}
                if self.val_loader is not None:
                    metrics, report = self.evaluate(self.val_loader)
                    logged.update({f"val/{key}": value for key, value in metrics.items()})
                    message += (
                        f" acc {metrics['accuracy']:.4f} macro_f1 {metrics['macro_f1']:.4f}"
                        f" f1_fake {metrics['f1_fake']:.4f}"
                    )
                    if metrics["macro_f1"] > best_score:
                        best_score = metrics["macro_f1"]
                        stale = 0
                        self._save("best.pt", epoch, metrics)
                        print(message)
                        print(report)
                    else:
                        stale += 1
                        if self.patience > 0:
                            message += f" stale {stale}/{self.patience}"
                        print(message)
                    stop = self.patience > 0 and stale >= self.patience
                else:
                    print(message)
                self.logger.log(logged, step=epoch)
                self._save("last.pt", epoch, metrics)
            if self._should_stop(stop):
                if self.is_main:
                    print(f"early stop at epoch {epoch}")
                break
        if self.is_main and self.val_loader is None:
            self._save("best.pt", epoch, None)
        if self.is_main and self.test_loader is not None:
            best_path = self.output_dir / "best.pt"
            if best_path.is_file():
                checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
                self._unwrap().load_state_dict(checkpoint["model"])
            metrics, report = self.evaluate(self.test_loader)
            self.logger.log({f"test/{key}": value for key, value in metrics.items()}, step=self.epochs)
            print("test", " ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
            print(report)
        if self.distributed:
            dist.barrier()

    def close(self) -> None:
        self.logger.finish()
        if dist.is_initialized():
            dist.destroy_process_group()


def _load_config(argv: list[str] | None = None) -> dict:
    config_name = "weibo"
    overrides: list[str] = []
    args = list(sys.argv[1:] if argv is None else argv)
    index = 0
    while index < len(args):
        arg = args[index]
        if arg.startswith("--local-rank") or arg.startswith("--local_rank"):
            index += 2 if "=" not in arg else 1
            continue
        if arg == "--config" and index + 1 < len(args):
            config_name = Path(args[index + 1]).stem
            index += 2
            continue
        if arg.startswith("--config="):
            config_name = Path(arg.split("=", 1)[1]).stem
            index += 1
            continue
        overrides.append(arg)
        index += 1
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name=config_name, overrides=overrides)
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise ValueError("Hydra config must be a mapping.")
    return container


def main() -> None:
    trainer = LoopFuseTrainer(_load_config())
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
