# Development Conventions

## Git Commit Rules

### When AI Assistant CAN Commit

- ❌ **NEVER** commit automatically
- ✅ **ONLY** when explicitly asked: "commit this" / "please commit" / "make a commit"
- ✅ **MUST** receive explicit permission for EVERY commit

### When AI Assistant CANNOT Commit

- ❌ After completing tasks/migrations
- ❌ When "finishing up" work
- ❌ At the end of conversations
- ❌ Without explicit user instruction

## Commit Message Format

**MUST** use Angular Conventional Commits style:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat`: new feature
- `fix`: bug fix
- `docs`: documentation only changes
- `style`: formatting, missing semi colons, etc
- `refactor`: code change that neither fixes a bug nor adds a feature
- `perf`: performance improvements
- `test`: adding missing tests
- `chore`: changes to build process or auxiliary tools
- `ci`: changes to CI configuration files and scripts

### Examples

```bash
feat(docker): add github workflow for automated builds
fix(handler): resolve openai compatibility issue
docs(readme): update installation instructions
refactor(engine): migrate from MODEL_PATH to MODEL_NAME
chore(deps): update requirements.txt
```

### Scope Guidelines

- Use component names: `docker`, `handler`, `engine`, `workflow`, `deps`
- Keep scopes short and descriptive
- Optional but recommended

## Code Quality

- Follow existing code style
- Test changes before committing
- Write descriptive commit messages
- Keep commits focused and atomic

## Configuration Conventions

- Single source of truth: use `.runpod/hub.json` for endpoint configuration.

  - Define environment variables, UI options, and allowed CUDA versions here.
  - Do not add or rely on `worker-config.json` (removed).

- CUDA policy:

  - **Custom base image**: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` with
    sglang >= 0.5.10 installed from pip.
  - CUDA 12.4 is required because Runpod Hub test VMs (RTX 4090) have NVIDIA drivers
    that only support up to CUDA 12.4. The upstream `lmsysorg/sglang` image for
    sglang >= 0.5.10 only ships with CUDA 13.0, which is rejected by
    `nvidia-container-cli` on these VMs.
  - Python 3.12 is installed from the deadsnakes PPA (Ubuntu 22.04 ships Python 3.10).
  - Two-tier GPU strategy:
    - **Test tier**: `ADA_24` (RTX 4090) + small model (`HuggingFaceTB/SmolLM2-1.7B-Instruct`)
      — validates basic worker functionality in Hub pre-release tests.
    - **Production tier**: `AMPERE_80` (A100 80GB) + large model (`Qwen/Qwen3.6-35B-A3B`)
      — actual inference deployment. Same image, different env vars.

- Tool/function calling and reasoning:
  - `TOOL_CALL_PARSER`: required to enable tool/function calling; no runtime default is applied. If unset, `--tool-call-parser` is not passed to SGLang.
  - `REASONING_PARSER`: required to enable reasoning trace parsing; no runtime default is applied. If unset, `--reasoning-parser` is not passed to SGLang.
  - Choose a parser matching the model family (e.g., `llama3`, `llama4`, `mistral`, `qwen25`, `deepseekv3`, `qwen3_coder`, `gpt-oss`, `kimi_k2`).
