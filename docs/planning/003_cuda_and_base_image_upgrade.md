# Decision Record: CUDA & Base Image Upgrade for Qwen3.6 Support

## Status

Accepted (2026-04-24)

## Context

The worker needs to support **Qwen/Qwen3.6-35B-A3B**, a Mixture-of-Experts model with
~35B total / ~3.6B active parameters. The [SGLang Qwen3.6 docs](https://cookbook.sglang.io/autoregressive/Qwen/Qwen3.6)
require **sglang >= 0.5.10** for this model.

### Current state

| Component    | Value                                    | Problem                                       |
|--------------|------------------------------------------|-----------------------------------------------|
| Base image   | `lmsysorg/sglang:v0.4.7.post1-cu124`     | SGLang 0.4.7 — too old for Qwen3.6           |
| CUDA variant | `cu124` (CUDA 12.4)                      | Compatible with test VMs but can't run Qwen3.6 |

### Available sglang >= 0.5.10 Docker tags

From Docker Hub (`lmsysorg/sglang`), v0.5.10+ only ships one CUDA variant:

| Tag                              | SGLang        | CUDA | Supports Qwen3.6?         |
|----------------------------------|---------------|------|---------------------------|
| `v0.4.7.post1-cu124`             | 0.4.7         | 12.4 | ❌                        |
| `v0.5.2-cu126`                   | 0.5.2         | 12.6 | ❌ (pre-Qwen3.6)          |
| **`v0.5.10.post1-cu130`**        | **0.5.10.post1** | **13.0** | **✅**              |

No cu124 or cu126 variant exists for sglang >= 0.5.10.

### nvidia-container-cli OCI prestart hook

Runpod Hub test VMs with RTX 4090 have NVIDIA drivers that only support CUDA 12.4.
When Docker starts a container with GPU support, `nvidia-container-cli` runs an OCI
prestart hook that checks the CUDA libraries inside the container against the host
NVIDIA driver version. If the container requires a higher CUDA version than the
driver supports, the container is rejected with:

```
nvidia-container-cli: unsatisfied condition: cuda>=13.0
```

This happens **before any process in the container starts** — it is not bypassable
from the Dockerfile, entrypoint, or environment variables.

## Decision

### 1. Custom base image: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` + pip sglang

Instead of using the official `lmsysorg/sglang:v0.5.10.post1-cu130` image (CUDA 13.0),
we build from `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` and install sglang from pip:

- **Base:** `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` — CUDA 12.4 runtime + cuDNN
- **Python:** 3.12 from deadsnakes PPA (Ubuntu 22.04 ships Python 3.10)
- **SGLang:** `pip install "sglang[all]>=0.5.10"` — pre-built CUDA 12.4 wheels
- **Package manager:** `uv` for pip-compatible package installation

This is the approach documented in the SGLang docs: "standard uv pip install sglang
path should work" for CUDA 12.4.

### 2. Two-tier GPU strategy

Runpod Hub tests use smaller GPUs with potentially older drivers. Production deployments
use larger GPUs with modern drivers. We handle this by:

| Tier         | GPU                      | VRAM | Model                                      | Purpose               |
|--------------|--------------------------|------|--------------------------------------------|-----------------------|
| **Test**     | RTX 4090 (`ADA_24`)      | 24 GB | `HuggingFaceTB/SmolLM2-1.7B-Instruct`       | Hub pre-release tests |
| **Production** | A100 80GB (`AMPERE_80`) | 80 GB | `Qwen/Qwen3.6-35B-A3B`                      | Actual inference      |

- RTX 4090 test VMs support CUDA 12.4 containers — compatible with our custom base image.
- A100 80GB GPUs on Runpod support both CUDA 12.4 and 13.0 — compatible either way.

### 3. Build/deploy workflow

Images are **never built or pushed locally**. The Runpod Serverless Hub handles this
automatically on release:

1. Push code changes to the `worker-sglang` repo
2. Create a Git tag (release) in the repo
3. Runpod Hub detects the release, builds the Docker image, runs Hub tests
4. On success, the image is available for endpoint deployment
5. Deploy to an endpoint with the desired GPU type and env vars

### 4. Override pattern

The same Docker image works for both tiers — environment variables control the model
and runtime settings:

**Hub template defaults** (in `hub.json`):

- `MODEL_NAME`: user-supplied (UI input)
- GPU: `ADA_24` (RTX 4090)

**Hub test defaults** (in `tests.json`):

- `MODEL_NAME`: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- GPU: `NVIDIA GeForce RTX 4090`

**Production override** (at endpoint creation):
- `MODEL_NAME`: `Qwen/Qwen3.6-35B-A3B`
- GPU: `NVIDIA A100 80GB`
- `CONTEXT_LENGTH`: `131072`
- `TOOL_CALL_PARSER`: `qwen3_coder`
- `REASONING_PARSER`: `qwen3`

## Consequences

### Positive
- Qwen3.6-35B-A3B runs with proper SGLang support
- Small model tests validate the worker independently of GPU size
- Same image, different env vars = flexible deployment
- CUDA 12.4 is compatible with Runpod test VMs (RTX 4090)
- No dependency on upstream `lmsysorg/sglang` Docker image releases

### Negative
- Larger Dockerfile — we now maintain the sglang installation ourselves
- Build time increases (pip installing sglang takes longer than using a pre-built image)
- Need to monitor sglang PyPI releases for updates and security patches

### Risk: flashinfer / CUDA wheel compatibility

`uv pip install "sglang[all]>=0.5.10"` installs pre-built wheels from PyPI including
flashinfer and flash-attn. These must be compatible with CUDA 12.4. As of sglang 0.5.10:

- SGLang publishes CUDA 12.4 wheels on PyPI
- flashinfer 0.1.x / 0.2.x supports CUDA 12.4
- flash-attn ships CUDA 12.4 wheels

If a future sglang release drops CUDA 12.4 wheel support, we pin to the last
compatible version or upgrade the CUDA base.

## Related

- [SGLang Qwen3.6 docs](https://cookbook.sglang.io/autoregressive/Qwen/Qwen3.6)
- [Qwen3.6-35B-A3B on HuggingFace](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [SGLang PyPI](https://pypi.org/project/sglang/)
- [SGLang GitHub: pip install from source docs](https://github.com/sgl-project/sglang)
