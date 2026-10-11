# Operator workbench (workbench/server.py + workbench/web/ + engine.py) as a container; laid out for a Hugging Face
# Docker Space (non-root uid 1000, writable HOME, app on port 7860).
#
#   docker build -t genco-workbench .
#   docker run --rm -p 7860:7860 genco-workbench            # http://localhost:7860
#   docker run --rm -e PORT=8000 -p 8000:8000 genco-workbench
#
# What is baked in at build time, so the container starts without any network access:
#
# * GENCO checkpoint (IEEE 118, Base, seed 0; 80 MB): DOWNLOADED from the Hugging Face model repo
#   gridfm/genco-pf-datakit (case118_ieee/base/seed0/{best_model_state_dict,normalizer_stats}.pt)
#   at a pinned revision and checked against the sha256 of the files this entry was developed with.
#   config.yaml is written by run_pretrained.config(), as `run_pretrained.py` does. The checkpoint is
#   not COPYed from the build context because work/pf_*/ is git-ignored and so absent from a Space
#   repository; downloading makes the build the same from a laptop and from a fresh clone.
#   To mount a checkpoint instead (folder with config.yaml, model/best_model_state_dict.pt,
#   model/normalizer_stats.pt, e.g. the one run_pretrained.py leaves in work/):
#       docker build --build-arg FETCH_CHECKPOINT=0 --build-arg WARMUP=0 -t genco-workbench .
#       docker run --rm -p 7860:7860 \
#           -v "$PWD/work/pf_case118_ieee_base:/home/user/app/work/pf_case118_ieee_base:ro" genco-workbench
#   (a mount also simply overrides the baked-in checkpoint of a default build).
# * Bus layout work/geo/case118.json: downloaded from pandapower v2.13.1 on GitHub (the URL
#   engine.layout() uses), sha256-checked.
# * Conformal band work/workbench_conformal_case118_ieee_base.json: COPYed from the repository
#   (2 kB, tracked in git), so the 4 MB sweep parquet it was calibrated on is not needed.
# * PGLib case pglib_opf_case118_ieee.m: gridfm-datakit downloads it into its own package folder on
#   first use and writes a "_corrected" copy next to it; triggered once during the build.
# * Julia: the official Julia binaries (pinned, from the julia Docker image) are put on PATH, where
#   juliapkg picks them up; importing juliacall then resolves the juliapkg.json files shipped with
#   juliacall and gridfm-datakit (PythonCall 0.9.36, PowerModels 0.21.6, Ipopt 1.15.0, Memento 1.5.0)
#   into PYTHON_JULIAPKG_PROJECT and precompiles them into JULIA_DEPOT_PATH. Afterwards
#   PYTHON_JULIAPKG_OFFLINE / JULIA_PKG_OFFLINE keep both from contacting a registry at start-up.
#   (The transitive Julia packages are whatever the registry resolves on the build day: the
#   repository has no Julia Manifest to pin them with.)
# * Python: requirements.lock.txt, with torch / torchvision / torchaudio taken from the PyTorch CPU
#   wheel index (no CUDA libraries) and torch-scatter compiled from its sdist against that torch
#   (PyPI has no wheels for it), which is why there is a builder stage with a compiler.
#
# The final build step starts the engine once and solves one state: the image fails to build
# rather than fail to start. Platforms: linux/amd64 and linux/arm64 (all pinned wheels exist for
# both; build natively - Julia under emulation is very slow).

ARG PYTHON_VERSION=3.12
ARG JULIA_VERSION=1.13.1
# Debian trixie on purpose: juliacall constrains Julia's OpenSSL_jll to at most the OpenSSL that
# Python is linked against, and Julia 1.13 needs OpenSSL 3.5 (trixie); bookworm has 3.0.

FROM julia:${JULIA_VERSION}-trixie AS julia

# ---------------------------------------------------------------- Python environment
FROM python:${PYTHON_VERSION}-slim-trixie AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/home/user/venv \
    PATH=/home/user/venv/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv $VIRTUAL_ENV

COPY requirements.lock.txt /tmp/requirements.lock.txt
# 1. CPU-only torch (the "+cpu" builds satisfy the "==" pins of the lock file)
# 2. torch-scatter from source: its setup.py imports torch, so no build isolation
# 3. everything else, torch and torch-scatter being already satisfied
RUN pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
        -c /tmp/requirements.lock.txt torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
    && pip install -c /tmp/requirements.lock.txt setuptools wheel \
    && FORCE_ONLY_CPU=1 pip install --no-build-isolation --no-deps torch-scatter==2.1.2 \
    && pip install -r /tmp/requirements.lock.txt \
    && python -c "import torch, torch_scatter; assert not torch.cuda.is_available(); print(torch.__version__, torch_scatter.__version__)" \
    && find $VIRTUAL_ENV -name __pycache__ -prune -exec rm -rf {} +

# ---------------------------------------------------------------- runtime image
FROM python:${PYTHON_VERSION}-slim-trixie

# libgomp: OpenMP runtime the compiled torch-scatter extension may link against
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user

COPY --from=julia /usr/local/julia /opt/julia
COPY --from=builder --chown=user:user /home/user/venv /home/user/venv

USER user
ENV HOME=/home/user \
    VIRTUAL_ENV=/home/user/venv \
    PATH=/home/user/venv/bin:/opt/julia/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JULIA_DEPOT_PATH=/home/user/.julia \
    PYTHON_JULIAPKG_PROJECT=/home/user/julia_env \
    JULIA_NUM_THREADS=1 \
    HF_HOME=/home/user/.cache/huggingface \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    XDG_CACHE_HOME=/home/user/.cache \
    GENCO_NETWORK=case118_ieee \
    GENCO_VARIANT=base \
    HOST=0.0.0.0 \
    PORT=7860
WORKDIR /home/user/app
RUN mkdir -p /home/user/.cache/matplotlib /home/user/.cache/huggingface work

# Julia environment + PGLib case. Depends only on the Python environment, so it stays cached
# when the application code changes.
COPY --chown=user:user workbench/docker/entrypoint.py workbench/docker/entrypoint.py
RUN python workbench/docker/entrypoint.py julia
# From here on nothing may reach for a Julia registry.
ENV PYTHON_JULIAPKG_OFFLINE=yes \
    JULIA_PKG_OFFLINE=true

# GENCO checkpoint, config.yaml and bus layout.
ARG FETCH_CHECKPOINT=1
ARG GENCO_REPO=gridfm/genco-pf-datakit
ARG GENCO_REVISION=7014f145635f26fa29e9e003f0dc33f689d615da
ARG GENCO_MODEL_SHA256=3ea2f552b190541fe08e0e27604daa5f122232af411f8bfd9bf355a557cab065
ARG GENCO_STATS_SHA256=01c93104fc47fbe4f7492d2926ad60fe7395f10f77f7788b2d71504ff6a68411
ARG GEO_SHA256=f3e46ed46fcfc07c531d5409227de7ed24712403401a1a25c778ab6897d87dcd
COPY --chown=user:user run_pretrained.py ./
RUN if [ "$FETCH_CHECKPOINT" = "1" ]; then \
        GENCO_REPO=$GENCO_REPO GENCO_REVISION=$GENCO_REVISION GENCO_MODEL_SHA256=$GENCO_MODEL_SHA256 \
        GENCO_STATS_SHA256=$GENCO_STATS_SHA256 GEO_SHA256=$GEO_SHA256 HF_HUB_DISABLE_TELEMETRY=1 \
        python workbench/docker/entrypoint.py fetch; \
    fi

# Application.
COPY --chown=user:user LICENSE engine.py ./
COPY --chown=user:user workbench/server.py workbench/
COPY --chown=user:user workbench/web/ workbench/web/
COPY --chown=user:user analysis/analysis_warmstart.py analysis/analysis_conformal.py analysis/
COPY --chown=user:user work/workbench_conformal_case118_ieee_base.json work/

# Smoke test: Julia, the checkpoint, AC-OPF, AC/DC power flow, GENCO and the staged check, offline settings on.
ARG WARMUP=1
RUN if [ "$WARMUP" = "1" ]; then HF_HUB_OFFLINE=1 python workbench/docker/entrypoint.py warmup; fi

ENV HF_HUB_OFFLINE=1
EXPOSE 7860
# The port only opens once Julia is up and the base case is dispatched (AC-OPF), hence the long start period.
HEALTHCHECK --interval=30s --timeout=20s --start-period=300s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/grid' % (os.environ.get('PORT') or 7860), timeout=15).read(1)" || exit 1

CMD ["python", "workbench/docker/entrypoint.py", "serve"]
