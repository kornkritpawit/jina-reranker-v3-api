FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip3 install --no-cache-dir --break-system-packages \
    fastapi "uvicorn[standard]" transformers pydantic python-dotenv && \
    pip3 install --no-cache-dir --break-system-packages \
    torch torchvision --index-url https://download.pytorch.org/whl/cu130

COPY app/ app/

EXPOSE ${PORT:-8000}

CMD ["sh", "-c", "python3 -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]