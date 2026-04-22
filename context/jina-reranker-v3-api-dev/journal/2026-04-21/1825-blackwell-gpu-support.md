# Blackwell GPU Support (RTX 5070 Ti)

**Date:** 2026-04-21 18:25 UTC
**Task:** Add Docker support for NVIDIA Blackwell architecture GPUs

## Problem

The standard Docker setup failed on a machine with NVIDIA RTX 5070 Ti (Blackwell architecture) with:

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12090).
Please update your GPU driver...
```

### Root Cause

- The standard `Dockerfile` installs PyTorch with CUDA 13.0 (`cu130`)
- RTX 5070 Ti (Blackwell, compute capability 12.0 / sm_120) requires CUDA 12.8+ toolkit
- The NVIDIA driver (575.64.03) supports CUDA 12.9, but PyTorch `cu130` requires a newer driver
- This creates a version mismatch between the PyTorch CUDA runtime and the host driver

### Machine Specs

- GPU: NVIDIA GeForce RTX 5070 Ti (Blackwell architecture)
- NVIDIA Driver: 575.64.03
- CUDA Version: 12.9
- OS: Linux

## Solution

Created separate Docker files that use PyTorch compiled with CUDA 12.8 (`cu128`) instead of CUDA 13.0 (`cu130`):

### Files Created

| File | Description |
|------|-------------|
| `Dockerfile.blackwell` | Dockerfile using `nvidia/cuda:12.8.0-runtime-ubuntu24.04` base + PyTorch `cu128` |
| `docker-compose.blackwell.yml` | Docker Compose config referencing `Dockerfile.blackwell` |

### Files Modified

| File | Change |
|------|--------|
| `context/docs/api.md` | Added "Blackwell GPU Support" section with usage instructions |

## Key Decision

- Kept the original `Dockerfile` unchanged for non-Blackwell GPU compatibility
- Used `cu128` PyTorch index URL: `https://download.pytorch.org/whl/cu128`
- Same base image (`nvidia/cuda:12.8.0-runtime-ubuntu24.04`) — only the PyTorch CUDA version differs

## Usage

```bash
# Build image สำหรับ Blackwell GPU
docker build -f Dockerfile.blackwell -t jina-reranker-v3-api-blackwell:latest .

# Run container พร้อม GPU
docker run -d \
  --gpus all \
  -p 8000:8000 \
  --env-file .env \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name reranker-blackwell \
  jina-reranker-v3-api-blackwell:latest

# ดู logs
docker logs -f reranker-blackwell

# หยุด container
docker stop reranker-blackwell && docker rm reranker-blackwell
```

> **หมายเหตุ:** หรือใช้ Docker Compose: `docker compose -f docker-compose.blackwell.yml up --build`

## Status

Completed — ready for testing on Blackwell GPU hardware.
