# Jina Reranker v3 API

FastAPI service for the [jinaai/jina-reranker-v3](https://huggingface.co/jinaai/jina-reranker-v3) model.

## Quick Start

### Local Setup

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA 13.0
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker Setup

```bash
# Build and run with GPU support
docker compose up --build

# Or build manually
docker build -t jina-reranker-v3-api .
docker run --gpus device=0 -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface jina-reranker-v3-api
```

## Usage

```bash
# Health check
curl http://localhost:8000/health

# Rerank documents
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "documents": ["ML is a subset of AI.", "The weather is sunny."],
    "top_n": 2,
    "return_documents": true
  }'
```

See [API Documentation](context/docs/api.md) for full details.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `jinaai/jina-reranker-v3` | HuggingFace model name |
| `DEVICE` | `cuda:0` | Torch device |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device(s) |

## Multi-Instance Setup (Load Balanced)

Run 3 reranker instances behind an nginx load balancer, all sharing a single GPU:

```bash
docker compose -f docker-compose.multi.yml up --build
```

This starts:
- **reranker-1** (port 8001), **reranker-2** (port 8002), **reranker-3** (port 8003) — loaded sequentially to avoid OOM
- **nginx** load balancer on port **8000** with round-robin across all 3 instances

All instances share the same GPU (`GPU_DEVICE_ID`, default `0`) and a shared HuggingFace cache volume.

```bash
# Test via load balancer
curl http://localhost:8000/health

curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "documents": ["ML is a subset of AI.", "The weather is sunny."],
    "top_n": 2
  }'
```

## Development

```bash
# Run tests
pytest tests/

# Lint
ruff check app/ tests/

# Type check
mypy app/