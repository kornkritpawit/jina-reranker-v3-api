# Jina Reranker v3 API Documentation

## Base URL

```
http://localhost:8000
```

## Endpoints

### Health Check

```
GET /health
```

**Response:**
```json
{"status": "ok"}
```

### Rerank

```
POST /v1/rerank
```

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | No | `jina-reranker-v3` | Model identifier |
| `query` | string | Yes | - | The query to rerank against |
| `documents` | array | Yes | - | List of strings or `{"text": "..."}` objects |
| `top_n` | integer | No | all | Number of top results to return |
| `return_documents` | boolean | No | `false` | Whether to include document text in response |

**Example Request:**
```json
{
  "query": "What is machine learning?",
  "documents": [
    "Machine learning is a subset of AI.",
    "The weather is sunny today.",
    "Deep learning uses neural networks."
  ],
  "top_n": 2,
  "return_documents": true
}
```

**Example Response:**
```json
{
  "model": "jina-reranker-v3",
  "usage": {
    "total_tokens": 45
  },
  "results": [
    {
      "index": 0,
      "relevance_score": 0.95,
      "document": {"text": "Machine learning is a subset of AI."}
    },
    {
      "index": 2,
      "relevance_score": 0.82,
      "document": {"text": "Deep learning uses neural networks."}
    }
  ]
}
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `jinaai/jina-reranker-v3` | HuggingFace model name |
| `DEVICE` | `cuda:0` | Torch device |
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `8000` | Server bind port |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device(s) to use |

## Blackwell GPU Support (RTX 5070 Ti / RTX 5080 / RTX 5090)

NVIDIA Blackwell architecture GPUs require PyTorch compiled with CUDA 12.8+. The standard `Dockerfile` uses CUDA 13.0 which is incompatible with Blackwell drivers.

### Requirements

- NVIDIA Driver >= 570.0
- CUDA 12.8+ compatible driver
- Docker with NVIDIA Container Toolkit

### Quick Start

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

### Files

| File | Description |
|------|-------------|
| `Dockerfile.blackwell` | Dockerfile using PyTorch cu128 for Blackwell GPUs |
| `docker-compose.blackwell.yml` | Docker Compose config that uses `Dockerfile.blackwell` (alternative method) |

### Key Difference

The Blackwell Dockerfile installs PyTorch from the `cu128` index instead of `cu130`:

```dockerfile
# Standard Dockerfile (non-Blackwell)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Blackwell Dockerfile
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Troubleshooting

If you see `RuntimeError: The NVIDIA driver on your system is too old`, ensure you are using `docker-compose.blackwell.yml` instead of the standard `docker-compose.yml`.