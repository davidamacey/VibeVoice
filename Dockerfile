# Stage 1: Compile flash-attn (needs nvcc from CUDA toolkit)
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS flash-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv \
    build-essential ninja-build git && \
    rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/build-venv
ENV PATH="/opt/build-venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir psutil && \
    pip install --no-cache-dir flash-attn --no-build-isolation

# Save the built wheel for copying
RUN pip wheel flash-attn --no-build-isolation --no-deps -w /tmp/wheels

# Stage 2: Runtime on Trixie
FROM python:3.12-slim-trixie

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch + CUDA runtime (pip pulls nvidia-* packages automatically)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

# Install pre-built flash-attn wheel from builder stage
COPY --from=flash-builder /tmp/wheels/*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

# ── Dependency layer (only rebuilds when pyproject.toml changes) ──────────────
# Copy only the manifest, then create minimal package stubs so that
# `pip install -e .` can discover and register the packages without needing the
# full source tree.  The real source is copied in the next layer (or mounted via
# docker-compose volumes at runtime), so source-file edits never invalidate this
# expensive layer.
COPY pyproject.toml .
RUN mkdir -p vibevoice vllm_plugin && \
    touch vibevoice/__init__.py vllm_plugin/__init__.py
RUN pip install --no-cache-dir torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir -e ".[streamingtts]" pytest "speechbrain>=1.0.0"

# ── Source layer (rebuilds on source changes, but pip install is already cached) ─
COPY vibevoice/ vibevoice/
COPY vllm_plugin/ vllm_plugin/
COPY demo/ demo/
COPY tests/ tests/

# NVIDIA container runtime mounts GPU drivers at runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "demo.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
