# Custom CUDA 12.4 base image for Runpod test VM compatibility.
# Runpod Hub test VMs (RTX 4090) support CUDA 12.4 but reject CUDA 13.0 containers
# due to nvidia-container-cli OCI prestart hook checking container CUDA libs vs host driver.
# sglang >= 0.5.10 is installed from pip (required for Qwen3.6-35B-A3B support).
ARG CUDA_BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_BASE_IMAGE}

# Install Python 3.12 from deadsnakes PPA (Ubuntu 22.04 ships Python 3.10)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    && rm -rf /var/lib/apt/lists/*

# Make python3 point to Python 3.12 (overrides Ubuntu 22.04 system python3.10)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install uv package manager
RUN curl -Ls https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv
ENV PATH="/root/.local/bin:${PATH}"

# Install sglang >= 0.5.10 (required for Qwen3.6-35B-A3B support).
# Ships pre-built wheels for CUDA 12.4 (flashinfer, flash-attn, etc.).
RUN uv pip install --system "sglang[all]>=0.5.10"

# Set working directory to the one already used by the base image
WORKDIR /sgl-workspace

# Optional transformers override for newer model architectures (e.g. qwen3_5_moe)
ARG TRANSFORMERS_SPEC=""
RUN if [ -n "${TRANSFORMERS_SPEC}" ]; then \
        uv pip install --system "${TRANSFORMERS_SPEC}"; \
    fi

# Install dependencies
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Copy source files
COPY handler.py engine.py utils.py download_model.py test_input.json ./
COPY public/ ./public/

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache" \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Model download script execution
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
        export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
        python3 download_model.py; \
    fi

CMD ["python3", "handler.py"]
