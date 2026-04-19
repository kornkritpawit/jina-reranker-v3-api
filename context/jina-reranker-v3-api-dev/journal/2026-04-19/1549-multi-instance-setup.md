# Multi-Instance Load Balanced Setup

**Time:** 2026-04-19 15:49 UTC
**Task:** Create multi-instance docker-compose with nginx load balancer

## Changes Made

1. **`docker-compose.multi.yml`** — 3 reranker instances (ports 8001-8003) behind nginx (port 8000), sequential startup via healthcheck dependencies, shared GPU and HF cache volume
2. **`nginx/nginx.conf`** — Round-robin upstream to 3 reranker backends
3. **`Dockerfile`** — Added `curl` for healthcheck, made port configurable via `PORT` env var
4. **`README.md`** — Documented multi-instance setup

## Architecture

```
Client → nginx:8000 → round-robin → reranker-1:8001
                                   → reranker-2:8002
                                   → reranker-3:8003
```

All 3 instances share a single GPU. Sequential startup (1→2→3) prevents OOM during model loading.