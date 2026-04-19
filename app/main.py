from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.config import API_KEY
from app.model import count_tokens, load_model, rerank

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    load_model()
    yield


app = FastAPI(title="Jina Reranker v3 API", lifespan=lifespan)

_bearer_scheme = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    if not API_KEY:
        return
    if credentials is None or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


class DocumentObject(BaseModel):
    text: str


class RerankRequest(BaseModel):
    model: str = "jina-reranker-v3"
    query: str
    documents: list[str | DocumentObject]
    top_n: int | None = None
    return_documents: bool = False


class DocumentResponse(BaseModel):
    text: str


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: DocumentResponse | None = None


class UsageInfo(BaseModel):
    total_tokens: int


class RerankResponse(BaseModel):
    model: str
    usage: UsageInfo
    results: list[RerankResult]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank_endpoint(
    request: RerankRequest, _auth: None = Depends(verify_api_key)
) -> RerankResponse:
    # Normalize documents to list of strings
    doc_strings: list[str] = []
    for doc in request.documents:
        if isinstance(doc, str):
            doc_strings.append(doc)
        else:
            doc_strings.append(doc.text)

    if not doc_strings:
        raise HTTPException(status_code=400, detail="documents must not be empty")

    try:
        results = rerank(request.query, doc_strings, top_n=request.top_n)
    except Exception as e:
        logger.exception("Rerank failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    total_tokens = count_tokens(request.query, doc_strings)

    rerank_results: list[RerankResult] = []
    for r in results:
        result = RerankResult(
            index=r["index"],
            relevance_score=r["relevance_score"],
        )
        if request.return_documents:
            result.document = DocumentResponse(text=doc_strings[r["index"]])
        rerank_results.append(result)

    return RerankResponse(
        model=request.model,
        usage=UsageInfo(total_tokens=total_tokens),
        results=rerank_results,
    )


if __name__ == "__main__":
    import uvicorn

    from app.config import HOST, PORT

    uvicorn.run(app, host=HOST, port=PORT)