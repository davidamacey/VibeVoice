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

# Install project
COPY pyproject.toml .
COPY vibevoice/ vibevoice/
COPY vllm_plugin/ vllm_plugin/
COPY demo/ demo/
COPY tests/ tests/
RUN pip install --no-cache-dir -e ".[streamingtts]" pytest

# NVIDIA container runtime mounts GPU drivers at runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8000

CMD ["python", "demo/realtime_model_inference_from_file.py", \
     "--model_path", "microsoft/VibeVoice-Realtime-0.5B", \
     "--timestamps", "--output_dir", "/app/outputs"]
