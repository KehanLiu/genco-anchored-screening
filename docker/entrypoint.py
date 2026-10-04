"""Container entrypoint of the workbench. server.py is left untouched: it binds 127.0.0.1 only,
so `serve` repeats what its main() does with the host and port taken from the environment.

    entrypoint.py julia     build time: resolve + precompile the Julia environment, cache the PGLib case
    entrypoint.py fetch     build time: GENCO checkpoint from Hugging Face, config.yaml, layout cache
    entrypoint.py warmup    build time: start the engine once and solve one state (smoke test)
    entrypoint.py [serve]   run time:   the web app on $HOST:$PORT (default 0.0.0.0:7860)
"""

import hashlib
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

NETWORK = os.environ.get("GENCO_NETWORK", "case118_ieee")
VARIANT = os.environ.get("GENCO_VARIANT", "base")

# Julia caches compiled package code for the CPU it was built on. The image is built on one machine
# and run on another, so compile for the same portable set of CPUs the official Julia binaries use.
# Has to be identical at build time and at run time, hence set here and not per command.
CPU_TARGETS = {
    "x86_64": "generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)",
    "aarch64": "generic;cortex-a57;thunderx2t99;carmel,clone_all;apple-m1,base(3);neoverse-512tvb,base(3)",
}
if platform.machine() in CPU_TARGETS:
    os.environ.setdefault("JULIA_CPU_TARGET", CPU_TARGETS[platform.machine()])


def check(path, sha256):
    """Fail the build if a downloaded file is not the one this image was written for."""
    found = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if sha256 and found != sha256:
        sys.exit(f"{path}: sha256 {found}, expected {sha256}")
    print(f"{path}: sha256 {found}")


def julia():
    # Importing juliacall (inside init_julia) makes juliapkg resolve the juliapkg.json files of
    # juliacall and gridfm-datakit (PythonCall, PowerModels, Ipopt, Memento) into
    # $PYTHON_JULIAPKG_PROJECT and precompile them into $JULIA_DEPOT_PATH.
    from gridfm_datakit.process.process_network import init_julia

    jl = init_julia(max_iter=200)
    try:  # `using PowerModels` above already compiled what the engine loads; this covers the rest
        jl.seval("using Pkg; Pkg.precompile(); Pkg.status()")
    except Exception as err:
        print("Pkg.precompile skipped:", err)
    # datakit downloads the PGLib case into its own package folder on first use and writes a
    # "_corrected" copy next to it (through a Julia subprocess); do both now.
    from gridfm_datakit.network import load_net_from_pglib

    load_net_from_pglib(NETWORK)


def fetch():
    import urllib.request

    import yaml
    from huggingface_hub import hf_hub_download

    import run_pretrained

    folder = ROOT / "work" / f"pf_{NETWORK}_{VARIANT}"
    (folder / "model").mkdir(parents=True, exist_ok=True)
    seed = os.environ.get("GENCO_SEED", "0")
    for name, sha in (("best_model_state_dict.pt", os.environ.get("GENCO_MODEL_SHA256")),
                      ("normalizer_stats.pt", os.environ.get("GENCO_STATS_SHA256"))):
        hf_hub_download(
            os.environ.get("GENCO_REPO", "gridfm/genco-pf-datakit"),
            f"{NETWORK}/{VARIANT}/seed{seed}/{name}",
            revision=os.environ.get("GENCO_REVISION", "main"),
            local_dir=folder / "hub",
        )
        (folder / "hub" / NETWORK / VARIANT / f"seed{seed}" / name).replace(folder / "model" / name)
        check(folder / "model" / name, sha)
    import shutil

    shutil.rmtree(folder / "hub")
    # the same config run_pretrained.py writes next to the checkpoint
    (folder / "config.yaml").write_text(yaml.safe_dump(run_pretrained.config(NETWORK, VARIANT, "cpu"), sort_keys=False))

    # bus coordinates: the file engine.layout() would otherwise download on first start
    case = NETWORK.removesuffix("_ieee")
    geo = ROOT / "work" / "geo" / f"{case}.json"
    if not geo.exists():
        geo.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/e2nIEE/pandapower/v2.13.1/"
            f"pandapower/networks/power_system_test_case_jsons/{case}.json",
            geo,
        )
    check(geo, os.environ.get("GEO_SHA256"))


def warmup():
    from engine import Engine

    engine = Engine(NETWORK, VARIANT)
    staged = engine.staged_info()
    print("staged check:", "no calibrated band" if staged is None else staged["widths"])
    result = engine.solve(1.0, [10], alpha=0.01)
    assert result["ok"], result
    print("warm-up solve ok:", result["timing_ms"], "stage", result["staged"] and result["staged"]["stage"])


def serve():
    # server imports engine first, which loads Julia before torch.
    import server
    import uvicorn
    from fastapi.staticfiles import StaticFiles

    # what server.main() does, with the host and port from the environment
    server.engine = server.Engine(NETWORK, VARIANT)
    server.app.mount("/", StaticFiles(directory=server.WEB, html=True), name="web")
    uvicorn.run(server.app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT") or 7860))


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "serve"
    {"julia": julia, "fetch": fetch, "warmup": warmup, "serve": serve}[command]()
