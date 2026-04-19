from __future__ import annotations

import logging
from typing import Any

from transformers import AutoModel, AutoTokenizer

from app.config import DEVICE, MODEL_NAME

logger = logging.getLogger(__name__)

_model: Any = None
_tokenizer: Any = None


def load_model() -> None:
    global _model, _tokenizer
    logger.info("Loading model %s on %s", MODEL_NAME, DEVICE)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _model = _model.to(DEVICE)
    _model.eval()
    logger.info("Model loaded successfully")


def get_model() -> Any:
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model


def get_tokenizer() -> Any:
    if _tokenizer is None:
        raise RuntimeError("Tokenizer not loaded. Call load_model() first.")
    return _tokenizer


def rerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    model = get_model()
    results: list[dict[str, Any]] = model.rerank(query, documents, top_n=top_n or len(documents))
    return results


def count_tokens(query: str, documents: list[str]) -> int:
    tokenizer = get_tokenizer()
    texts = [query, *documents]
    total = 0
    for text in texts:
        total += len(tokenizer.encode(text, add_special_tokens=False))
    return total