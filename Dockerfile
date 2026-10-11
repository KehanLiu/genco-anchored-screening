# Operator workbench (workbench/server.py + workbench/web/ + engine.py) as a container; laid out for a Hugging Face
# Docker Space (non-root uid 1000, writable HOME, app on port 7860).
#
#   docker build -t genco-workbench .
#   docker run --rm -p 7860:7860 genco-workbench            # http://localhost:7860
#   docker run --rm -e PORT=8000 -p 8000:8000 genco-workbench
#
# Verified build (2026-10-11, Apple Silicon, Docker Desktop 29.7.2, built natively for linux/arm64):
#   docker build --platform linux/arm64 -t genco-workbench:dev .
#   docker run --rm -p 7860:7860 genco-workbench:dev
#   Build time 50 min 28 s with a cold pip cache, of which 44 min were wheel downloads through Docker
#   Desktop's proxy (torch step 797 s, requirements.lock.txt 1837 s); torch-scatter compiled from its
#   sdist in 95 s, the Julia environment resolved and precompiled in 193 s, warm-up solve 20 s.
#   Image: 4.81 GB on disk, 1.07 GB compressed (docker images). The server answers ~16 s after start.
#   Smoke test: GET / -> 200 (4276 B html), GET /api/grid -> 200 (118 buses, 186 branches, staged band),
#   POST /api/solve {"load_scale":1.1,"outages":[31,103],"alpha":0.01} -> 200, ok=true, stage 3,
#   verdict "overloaded" decided by the AC check, timing_ms {ac: 2.9, dc: 1.3, genco: 89}.
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
#   (data.pyg.org/whl/torch-2.13.0+cpu lists only pyg_lib, no torch_scatter, so the sdist build stays.)
# * gridtrust/src/gridtrust: imported by analysis_conformal.py, which engine.conformal_band() only
#   loads when the band JSON is missing; copied so that path works too.
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

# apt over https with retries: plain-http connections to deb.debian.org failed intermittently from
# BuildKit's build network on Docker Desktop (both stages, two builds in a row).
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# pip's download cache lives in a BuildKit cache mount (not in the image), so a build retried after a
# flaky download picks up where it left off. Retries/timeout: downloads through Docker Desktop's proxy
# stalled and once arrived corrupt (hash mismatch).
ENV VIRTUAL_ENV=/home/user/venv \
    PATH=/home/user/venv/bin:$PATH \
    PIP_CACHE_DIR=/root/.cache/pip \
    PIP_RETRIES=10 \
    PIP_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv $VIRTUAL_ENV

COPY requirements.lock.txt /tmp/requirements.lock.txt
# 1. CPU-only torch (the "+cpu" builds satisfy the "==" pins of the lock file)
# 2. torch-scatter from source: its setup.py imports torch, so no build isolation
# 3. everything else, torch and torch-scatter being already satisfied
# (three RUN steps so that a failure in a later one keeps the earlier layers)
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
        -c /tmp/requirements.lock.txt torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install -c /tmp/requirements.lock.txt setuptools wheel \
    && FORCE_ONLY_CPU=1 pip install --no-build-isolation --no-deps torch-scatter==2.1.2
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install -r /tmp/requirements.lock.txt \
    && python -c "import torch, torch_scatter; assert not torch.cuda.is_available(); print(torch.__version__, torch_scatter.__version__)" \
    && find $VIRTUAL_ENV -name __pycache__ -prune -exec rm -rf {} +

# ---------------------------------------------------------------- runtime image
FROM python:${PYTHON_VERSION}-slim-trixie

# libgomp: OpenMP runtime the compiled torch-scatter extension may link against
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends libgomp1 ca-certificates \
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
# analysis_conformal.py imports gridtrust from ../gridtrust/src; only reached when the band JSON below is absent.
COPY --chown=user:user gridtrust/src/gridtrust gridtrust/src/gridtrust
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
