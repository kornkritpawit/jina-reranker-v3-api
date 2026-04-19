from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


TEST_API_KEY = "test-secret-key"


@pytest.fixture()
def client():
    with patch("app.main.API_KEY", TEST_API_KEY), patch("app.main.load_model"):
        from app.main import app

        with TestClient(app) as c:
            yield c


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TEST_API_KEY}"}


def test_health(client: TestClient):
    response = client.get("/health")  # health has no auth
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("app.main.rerank")
@patch("app.main.count_tokens", return_value=42)
def test_rerank_basic(mock_tokens: MagicMock, mock_rerank: MagicMock, client: TestClient):
    mock_rerank.return_value = [
        {"index": 1, "relevance_score": 0.9, "document": {"text": "doc2"}},
        {"index": 0, "relevance_score": 0.5, "document": {"text": "doc1"}},
    ]
    response = client.post(
        "/v1/rerank",
        headers=_auth_headers(),
        json={
            "query": "test query",
            "documents": ["doc1", "doc2"],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "jina-reranker-v3"
    assert data["usage"]["total_tokens"] == 42
    assert len(data["results"]) == 2
    assert data["results"][0]["relevance_score"] == 0.9
    assert data["results"][0].get("document") is None


@patch("app.main.rerank")
@patch("app.main.count_tokens", return_value=30)
def test_rerank_return_documents(
    mock_tokens: MagicMock, mock_rerank: MagicMock, client: TestClient
):
    mock_rerank.return_value = [
        {"index": 0, "relevance_score": 0.8, "document": {"text": "doc1"}},
    ]
    response = client.post(
        "/v1/rerank",
        headers=_auth_headers(),
        json={
            "query": "test query",
            "documents": ["doc1", "doc2"],
            "return_documents": True,
            "top_n": 1,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["document"]["text"] == "doc1"


@patch("app.main.rerank")
@patch("app.main.count_tokens", return_value=20)
def test_rerank_with_document_objects(
    mock_tokens: MagicMock, mock_rerank: MagicMock, client: TestClient
):
    mock_rerank.return_value = [
        {"index": 0, "relevance_score": 0.7, "document": {"text": "hello"}},
    ]
    response = client.post(
        "/v1/rerank",
        headers=_auth_headers(),
        json={
            "query": "test",
            "documents": [{"text": "hello"}, {"text": "world"}],
            "top_n": 1,
        },
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == 1


def test_rerank_unauthorized(client: TestClient):
    response = client.post(
        "/v1/rerank",
        json={"query": "test", "documents": ["doc1"]},
    )
    assert response.status_code == 401


def test_rerank_wrong_token(client: TestClient):
    response = client.post(
        "/v1/rerank",
        headers={"Authorization": "Bearer wrong-key"},
        json={"query": "test", "documents": ["doc1"]},
    )
    assert response.status_code == 401