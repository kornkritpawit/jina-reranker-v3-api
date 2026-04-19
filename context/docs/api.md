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