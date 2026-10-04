"""Step-budgeted, resumable fine-tuning of a released GENCO checkpoint (graphkit task / datamodule / losses unchanged).

Why not `gridfm_graphkit finetune`: the CLI is epoch-based, cannot resume, and re-fits the normalizer (baseMVA =
95th percentile of the training data) instead of keeping the released one. This script uses the same PowerFlowTask,
LitGridHeteroDataModule and released hyper-parameters (released_training.yaml, converted in memory), but
  - restores the RELEASED normalizer stats (so the fine-tuned weights stay a drop-in for engine.Genco),
  - takes explicit train/val/test scenario lists (train.pt / val.pt / test.pt in --splits),
  - stops after --max_steps optimizer steps, validates + checkpoints every --val_every steps,
  - resumes from <out>/ckpt/last.ckpt if present (optimizer state, step count, best validation loss).
Output (<out> = work/finetune/models/pf_<net>_<tag>, loadable by engine.Genco with engine.WORK = work/finetune/models):
  config.yaml, model/best_model_state_dict.pt (lowest validation loss), model/last_model_state_dict.pt,
  model/normalizer_stats.pt, train_log.csv (step, seconds, train loss mean since last check, validation loss, lr).

    work/texas/tmo 7200 .venv/bin/python -u work/finetune/train.py --net case500_goc --data work/finetune/data500_ft \
        --splits work/finetune/data500_ft/splits_base --init work/goc500/pf_case500_goc_tiny --tag ft_seed0 \
        --max_steps 6000 --val_every 500 --accelerator cpu
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path(__file__).resolve().parent / "models"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--net", required=True)
    p.add_argument("--data", required=True, help="folder containing <net>/raw")
    p.add_argument("--splits", required=True, help="folder with train.pt / val.pt / test.pt (scenario ids)")
    p.add_argument("--init", required=True, help="folder with released_training.yaml? config.yaml, model/*.pt")
    p.add_argument("--hparams", default="", help="released_training.yaml (default: <init>/released_training.yaml)")
    p.add_argument("--tag", required=True)
    p.add_argument("--max_steps", type=int, required=True)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--accelerator", default="cpu")
    p.add_argument("--lr", type=float, default=0.0, help="0 = released value")
    p.add_argument("--batch", type=int, default=0, help="0 = released value")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit_val", type=int, default=0, help="max validation batches (0 = all)")
    a = p.parse_args()

    import lightning as L
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
    from gridfm_graphkit.io.config_version import resolve_config
    from gridfm_graphkit.io.param_handler import NestedNamespace, get_task

    torch.set_num_threads(a.threads)
    init = Path(a.init)
    out = MODELS / f"pf_{a.net}_{a.tag}"
    (out / "model").mkdir(parents=True, exist_ok=True)
    hp = Path(a.hparams) if a.hparams else init / "released_training.yaml"
    cfg = resolve_config(yaml.safe_load(hp.read_text()), config_path=str(hp), conversion_mode="auto_inline", converted_config_path=None)
    cfg["data"].update(workers=a.workers, split_by_load_scenario_idx=False, split_from_existing_files=str(a.splits), stream_partitions="off")
    cfg["training"].update(accelerator=a.accelerator, devices=1, strategy="auto")
    if a.lr:
        cfg["optimizer"]["learning_rate"] = a.lr
    if a.batch:
        cfg["training"]["batch_size"] = a.batch
    cfg["seed"] = a.seed
    (out / "train_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    args = NestedNamespace(**cfg)
    L.seed_everything(a.seed, workers=True)

    stats = init / "model" / "normalizer_stats.pt"
    dm = LitGridHeteroDataModule(args, a.data, normalizer_stats_path=str(stats))
    model = get_task(args, dm.data_normalizers)
    ckpt = out / "ckpt" / "last.ckpt"
    start = init / "model" / "best_model_state_dict.pt"
    model.load_state_dict(torch.load(start, map_location="cpu"))
    print(f"init weights {start}; normalizer {torch.load(stats, weights_only=True)}; resume from {ckpt if ckpt.exists() else 'nothing'}", flush=True)

    class Track(Callback):
        def __init__(self):
            self.best = float("inf")
            self.losses, self.t0, self.s0, self.spent = [], None, None, 0.0
            self.log = out / "train_log.csv"

        def state_dict(self):
            return {"best": self.best, "spent": self.spent}

        def load_state_dict(self, s):
            self.best, self.spent = s["best"], s.get("spent", 0.0)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if self.t0 is None:
                self.t0, self.s0 = time.perf_counter(), trainer.global_step
            self.losses.append(float(outputs["loss"]))
            if not torch.isfinite(outputs["loss"]):
                raise RuntimeError(f"non-finite training loss at step {trainer.global_step}")

        def on_validation_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            val = float(trainer.callback_metrics["Validation loss"])
            now = time.perf_counter()
            dt = now - (self.t0 or now)
            n = trainer.global_step - (self.s0 or trainer.global_step)
            self.spent += dt
            tr = sum(self.losses) / max(len(self.losses), 1)
            sd = {k.replace("model._orig_mod.", "model."): v.detach().cpu() for k, v in pl_module.state_dict().items()}
            torch.save(sd, out / "model" / "last_model_state_dict.pt")
            best = val < self.best
            if best:
                self.best = val
                torch.save(sd, out / "model" / "best_model_state_dict.pt")
            lr = pl_module.optimizer.param_groups[0]["lr"]
            if not self.log.exists():
                self.log.write_text("step,train_seconds,sec_per_step,train_loss,val_loss,lr,best\n")
            with self.log.open("a") as f:
                f.write(f"{trainer.global_step},{self.spent:.1f},{dt / max(n, 1):.4f},{tr:.6g},{val:.6g},{lr:.3g},{int(best)}\n")
            print(time.strftime("%H:%M:%S"), f"step {trainer.global_step}: train loss {tr:.5g}, val loss {val:.5g}{' *' if best else ''}, "
                  f"{dt / max(n, 1):.3f} s/step ({n} steps), lr {lr:.2g}", flush=True)
            self.losses, self.t0, self.s0 = [], None, None

    trainer = L.Trainer(
        logger=CSVLogger(str(out), name="csv"),
        accelerator=a.accelerator, devices=1, strategy="auto",
        max_steps=a.max_steps, max_epochs=-1,
        val_check_interval=a.val_every, check_val_every_n_epoch=None,
        limit_val_batches=a.limit_val or 1.0,
        num_sanity_val_steps=0, log_every_n_steps=100, enable_progress_bar=False,
        default_root_dir=str(out),
        callbacks=[Track(), ModelCheckpoint(dirpath=str(out / "ckpt"), save_last=True, save_top_k=0, every_n_train_steps=a.val_every)],
    )
    t = time.perf_counter()
    trainer.fit(model=model, datamodule=dm, ckpt_path=str(ckpt) if ckpt.exists() else None)
    print(f"done: global step {trainer.global_step}, wall {time.perf_counter() - t:.0f}s this launch; "
          f"train {len(dm.train_dataset_multi)} / val {len(dm.val_dataset_multi)} scenarios, batch {args.training.batch_size}", flush=True)
    shutil.copyfile(stats, out / "model" / "normalizer_stats.pt")
    shutil.copyfile(init / "config.yaml", out / "config.yaml")
    (out / "DONE").write_text(time.strftime("%F %T") + f" step {trainer.global_step}\n")


if __name__ == "__main__":
    main()
