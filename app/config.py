import os

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME: str = os.getenv("MODEL_NAME", "jinaai/jina-reranker-v3")
DEVICE: str = os.getenv("DEVICE", "cuda:0")
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
API_KEY: str = os.getenv("API_KEY", "")