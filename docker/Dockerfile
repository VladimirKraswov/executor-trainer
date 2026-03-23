# Base image with CUDA and python
ARG BASE_IMAGE=igortet/model-qwen-7b
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    VLLM_PYTHON_BIN=/opt/vllm-venv/bin/python

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    bash \
    ca-certificates \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /trainer

# Copy requirement files first for better caching
COPY requirements.txt requirements.vllm.txt /trainer/

# Copy local wheels if any (empty if none)
COPY packages /trainer/packages
COPY docker/install_with_local_wheels.sh /trainer/install_with_local_wheels.sh
RUN chmod +x /trainer/install_with_local_wheels.sh

# Setup training virtual environment
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install -U pip setuptools wheel && \
    /trainer/install_with_local_wheels.sh /opt/venv/bin/python --requirements /trainer/requirements.txt /trainer/packages/pip

# Setup separate vLLM virtual environment
RUN python3 -m venv /opt/vllm-venv && \
    /opt/vllm-venv/bin/pip install -U pip setuptools wheel && \
    /trainer/install_with_local_wheels.sh /opt/vllm-venv/bin/python --requirements /trainer/requirements.vllm.txt /trainer/packages/pip

# Copy application code
COPY app /trainer/app
COPY docker/entrypoint.sh /trainer/entrypoint.sh
RUN chmod +x /trainer/entrypoint.sh

# Final setup
RUN mkdir -p /output /cache/huggingface && \
    chmod -R 777 /output /cache/huggingface

ENTRYPOINT ["/trainer/entrypoint.sh"]
