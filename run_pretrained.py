"""Run a pretrained GENCO power-flow checkpoint end to end on one data shard.

Downloads the first partition (200 scenarios) of a gridfm PF dataset and one
GENCO checkpoint from Hugging Face, runs `gridfm_graphkit predict`, and compares
the predictions against the AC solution stored in the dataset.

    .venv/bin/python run_pretrained.py --network case14_ieee --variant tiny
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from huggingface_hub import hf_hub_download, list_repo_files

HIDDEN_SIZE = {"tiny": 12, "small": 24, "base": 48}
TABLES = ("bus_data.parquet", "gen_data.parquet", "branch_data.parquet")
WORK = Path(__file__).parent / "work"


def place(repo_id, filename, dest, repo_type=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copyfile(hf_hub_download(repo_id, filename, repo_type=repo_type), dest)
    return dest


def config(network, variant, accelerator):
    return {
        "task": {"task_name": "PowerFlow"},
        "data": {
            "baseMVA": 100,
            "mask_value": 0.0,
            "normalization": "HeteroDataMVANormalizer",
            "networks": [network],
            "scenarios": [200],
            "test_ratio": 0.15,
            "val_ratio": 0.1,
            "workers": 0,
            "split_by_load_scenario_idx": True,
        },
        "model": {
            "attention_head": 8,
            "edge_dim": 10,
            "hidden_size": HIDDEN_SIZE[variant],
            "input_bus_dim": 15,
            "input_gen_dim": 6,
            "output_bus_dim": 2,
            "output_gen_dim": 1,
            "num_layers": 12,
            "type": "GNS_heterogeneous",
        },
        "optimizer": {
            "beta1": 0.9,
            "beta2": 0.999,
            "learning_rate": 0.0005,
            "lr_decay": 0.7,
            "lr_patience": 5,
        },
        "training": {
            "batch_size": 8,
            "epochs": 1,
            "loss_weights": [0.1, 0.9],
            "losses": ["LayeredWeightedPhysics", "MaskedBusMSE"],
            "loss_args": [{"base_weight": 0.5}, {}],
            "accelerator": accelerator,
            "devices": 1,
            "strategy": "auto",
        },
        "seed": 0,
        "verbose": False,
        "callbacks": {"patience": 100, "tol": 0},
    }


def prepare(network, variant, seed, accelerator):
    # Seed 0 keeps the plain folder name the engine loads by default; other seeds get a suffix.
    work = WORK / (f"pf_{network}_{variant}" + (f"_seed{seed}" if seed else ""))
    raw = work / "data" / network / "raw"
    data_repo = f"gridfm/pf_small_{network}"
    remote_files = list_repo_files(data_repo, repo_type="dataset")
    for table in TABLES:
        prefix = f"{table}/scenario_partition=0/"
        for remote in (f for f in remote_files if f.startswith(prefix)):
            place(data_repo, remote, raw / remote, repo_type="dataset")
    prefix = f"{network}/{variant}/seed{seed}"
    model = place("gridfm/genco-pf-datakit", f"{prefix}/best_model_state_dict.pt", work / "model" / "best_model_state_dict.pt")
    stats = place("gridfm/genco-pf-datakit", f"{prefix}/normalizer_stats.pt", work / "model" / "normalizer_stats.pt")
    config_path = work / "config.yaml"
    config_path.write_text(yaml.safe_dump(config(network, variant, accelerator), sort_keys=False))
    return work, model, stats, config_path


def predict(work, model, stats, config_path):
    out = work / "out"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    cmd = [
        str(Path(sys.executable).parent / "gridfm_graphkit"), "predict",
        "--config", str(config_path),
        "--config_conversion", "auto_inline",
        "--model_path", str(model),
        "--normalizer_stats", str(stats),
        "--data_path", str(work / "data"),
        "--log_dir", str(work / "mlruns"),
        "--output_path", str(out),
        "--batch_size", "8",
        "--num_workers", "0",
    ]
    start = time.perf_counter()
    subprocess.run(cmd, check=True, env={**os.environ, "MLFLOW_ALLOW_FILE_STORE": "true"})
    return out, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="case14_ieee")
    parser.add_argument("--variant", default="tiny", choices=HIDDEN_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--accelerator", default="cpu", choices=["cpu", "mps"])
    args = parser.parse_args()

    work, model, stats, config_path = prepare(args.network, args.variant, args.seed, args.accelerator)
    out, seconds = predict(work, model, stats, config_path)

    print(f"\npredict wall time (incl. process start and data processing): {seconds:.1f}s")
    summarize(out, args.network)


def summarize(out, network):
    bus = pd.read_parquet(out / f"{network}_predictions.parquet")
    branch = pd.read_parquet(out / f"{network}_branch_predictions.parquet")
    # Branch rows are directed (one per direction) and vary with line outages.
    print(f"{bus.scenario.nunique()} scenarios, {bus.bus.nunique()} buses, {len(branch)} directed branch rows")

    # Only the quantities the solver has to find: Vm at load buses, Va away from
    # the reference bus, Pg at the reference bus, Qg at generator buses.
    errors = {
        "Vm at PQ buses (p.u.)": (bus.Vm_pred - bus.Vm_target)[bus.PQ == 1],
        "Va at non-REF buses (deg)": np.degrees(bus.Va_pred - bus.Va_target)[bus.REF == 0],
        "Pg at REF bus (MW)": (bus.Pg_pred - bus.Pg_target)[bus.REF == 1],
        "Qg at PV/REF buses (MVAr)": (bus.Qg_pred - bus.Qg_target)[bus.PQ == 0],
        "branch P flow (MW)": branch.Pft - branch.Pft_target,
        "branch Q flow (MVAr)": branch.Qft - branch.Qft_target,
        "active power-balance residual (MW)": bus["active res. (MW)"],
        "reactive power-balance residual (MVAr)": bus["reactive res. (MVar)"],
    }
    print(f"{'quantity':<42}{'mean abs':>12}{'max abs':>12}")
    for name, err in errors.items():
        print(f"{name:<42}{err.abs().mean():>12.5f}{err.abs().max():>12.5f}")


if __name__ == "__main__":
    main()
