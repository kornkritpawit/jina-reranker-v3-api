#!/bin/bash
# Set API_KEY if provided as env var or first argument
API_KEY="${API_KEY:-$1}"
AUTH_HEADER=""
if [ -n "$API_KEY" ]; then
  AUTH_HEADER="-H \"Authorization: Bearer $API_KEY\""
fi

# Test health endpoint
echo "=== Health Check ==="
curl -s http://localhost:8000/health | python3 -m json.tool

echo ""
echo "=== Rerank (basic) ==="
eval curl -s -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  ${AUTH_HEADER} \
  -d "'"'{
    "query": "What is machine learning?",
    "documents": [
      "Machine learning is a subset of artificial intelligence.",
      "The weather today is sunny.",
      "Deep learning uses neural networks with many layers."
    ]
  }'"'" | python3 -m json.tool

echo ""
echo "=== Rerank (with return_documents and top_n) ==="
eval curl -s -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  ${AUTH_HEADER} \
  -d "'"'{
    "query": "What is machine learning?",
    "documents": [
      "Machine learning is a subset of artificial intelligence.",
      "The weather today is sunny.",
      "Deep learning uses neural networks with many layers."
    ],
    "top_n": 2,
    "return_documents": true
  }'"'" | python3 -m json.tool