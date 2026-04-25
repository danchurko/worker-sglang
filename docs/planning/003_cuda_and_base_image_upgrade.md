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
| CUDA variant | `cu124` (CUDA 12.4)                      | Compatible with older drivers but can't run Qwen3.6 |

### Available sglang >= 0.5.10 Docker tags

From Docker Hub (`lmsysorg/sglang`), v0.5.10+ only ships one CUDA variant:

| Tag                              | SGLang        | CUDA | Supports Qwen3.6?         |
|----------------------------------|---------------|------|---------------------------|
| `v0.4.7.post1-cu124`             | 0.4.7         | 12.4 | ❌                        |
| `v0.5.2-cu126`                   | 0.5.2         | 12.6 | ❌ (pre-Qwen3.6)          |
| **`v0.5.10.post1-cu130`**        | **0.5.10.post1** | **13.0** | **✅**              |

No cu124 or cu126 variant exists for sglang >= 0.5.10.

## Decision

### 1. Upgrade base image to `lmsysorg/sglang:v0.5.10.post1-cu130`

This is the minimum version that supports Qwen3.6-35B-A3B. The `latest` tag resolves to
this same image (`v0.5.10.post1-cu130`).

### 2. Two-tier GPU strategy

Runpod Hub tests use smaller GPUs with potentially older drivers. Production deployments
use larger GPUs with modern drivers. We handle this by:

| Tier         | GPU                      | VRAM | Model                                      | Purpose               |
|--------------|--------------------------|------|--------------------------------------------|-----------------------|
| **Test**     | RTX 4090 (`ADA_24`)      | 24 GB | `HuggingFaceTB/SmolLM2-1.7B-Instruct`       | Hub pre-release tests |
| **Production** | A100 80GB (`AMPERE_80`) | 80 GB | `Qwen/Qwen3.6-35B-A3B`                      | Actual inference      |

- A100 80GB GPUs on Runpod run modern NVIDIA drivers that support CUDA 13.0.
- RTX 4090 test VMs may have older drivers — the small model test validates basic
  container functionality (startup, HTTP API, response format) without relying on
  model-specific CUDA features.
- If the CUDA 13.0 requirement blocks RTX 4090 tests entirely, the fallback is to
  build a custom base image with sglang 0.5.10+ compiled against CUDA 12.4/12.6.

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

### Negative
- CUDA 13.0 requires newer NVIDIA drivers — may not be available on all Runpod test VMs
- RTX 4090 tests may fail if the host driver doesn't support CUDA 13.0 containers

### Mitigation
- If Hub tests fail on RTX 4090, switch test GPU to a model with newer drivers
  (e.g., L40S, A100) or build a custom base image with CUDA 12.4 + sglang >= 0.5.10
  from source

## Related

- [SGLang Qwen3.6 docs](https://cookbook.sglang.io/autoregressive/Qwen/Qwen3.6)
- [Qwen3.6-35B-A3B on HuggingFace](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- SGLang releases: https://github.com/sgl-project/sglang/releases
