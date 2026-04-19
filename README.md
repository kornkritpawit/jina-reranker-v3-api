# Jina Reranker v3 API

FastAPI service สำหรับโมเดล [jinaai/jina-reranker-v3](https://huggingface.co/jinaai/jina-reranker-v3)

## Quick Start

### ติดตั้งแบบ Local

```bash
# สร้าง virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# ติดตั้ง PyTorch พร้อม CUDA 13.0
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# ติดตั้ง dependencies
pip install -e ".[dev]"

# คัดลอกและตั้งค่า environment
cp .env.example .env

# รันเซิร์ฟเวอร์
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Docker Build & Run

### ข้อกำหนดเบื้องต้น

- Docker Engine พร้อม [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- GPU ที่รองรับ CUDA
- Docker Compose v2 (สำหรับการใช้งานแบบ compose)

### ตั้งค่า Environment Variables

คัดลอกไฟล์ `.env.example` เป็น `.env` แล้วแก้ไขตามต้องการ:

```bash
cp .env.example .env
```

| ตัวแปร | ค่าเริ่มต้น | คำอธิบาย |
|--------|-------------|----------|
| `MODEL_NAME` | `jinaai/jina-reranker-v3` | ชื่อโมเดลจาก HuggingFace |
| `DEVICE` | `cuda:0` | อุปกรณ์ Torch ที่ใช้ |
| `HOST` | `0.0.0.0` | Host ของเซิร์ฟเวอร์ |
| `PORT` | `8000` | Port ของเซิร์ฟเวอร์ |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU ที่ต้องการใช้ |
| `GPU_DEVICE_ID` | `0` | GPU device ID สำหรับ Docker deploy (เช่น `"0"`, `"1"`, `"0,1"`) |
| `API_KEY` | _(ว่าง)_ | API key สำหรับ authentication (ถ้ามี) |

### วิธีที่ 1: Build และ Run ด้วย Docker โดยตรง

```bash
# Build image
docker build -t jina-reranker-v3-api:latest .

# Run container พร้อม GPU
docker run -d \
  --gpus device=0 \
  -p 8000:8000 \
  -e MODEL_NAME=jinaai/jina-reranker-v3 \
  -e DEVICE=cuda:0 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name reranker \
  jina-reranker-v3-api

# ดู logs
docker logs -f reranker

# หยุด container
docker stop reranker && docker rm reranker
```

> **หมายเหตุ:** Volume mount `-v ~/.cache/huggingface:/root/.cache/huggingface` ใช้เพื่อ cache โมเดลไว้ในเครื่อง host ไม่ต้องดาวน์โหลดใหม่ทุกครั้ง

### วิธีที่ 2: Docker Compose (Single Instance)

ใช้ไฟล์ `docker-compose.yml` สำหรับรัน instance เดียว:

```bash
# Build และ run
docker compose up --build

# Run แบบ background
docker compose up --build -d

# หยุด
docker compose down
```

กำหนด GPU ที่ต้องการใช้ผ่านตัวแปร `GPU_DEVICE_ID`:

```bash
GPU_DEVICE_ID=1 docker compose up --build -d
```

### วิธีที่ 3: Docker Compose Multi-Instance (Load Balanced)

ใช้ไฟล์ `docker-compose.multi.yml` สำหรับรัน 3 instances พร้อม nginx load balancer:

```bash
# Build และ run ทั้ง 3 instances + nginx
docker compose -f docker-compose.multi.yml up --build

# Run แบบ background
docker compose -f docker-compose.multi.yml up --build -d

# หยุดทั้งหมด
docker compose -f docker-compose.multi.yml down
```

สถาปัตยกรรม:
- **reranker-1** (port 8001), **reranker-2** (port 8002), **reranker-3** (port 8003) — โหลดตามลำดับเพื่อป้องกัน OOM
- **nginx** load balancer บน port **8000** ด้วย round-robin
- ทุก instance ใช้ GPU เดียวกัน (`GPU_DEVICE_ID`, ค่าเริ่มต้น `0`) และแชร์ HuggingFace cache volume

---

## การใช้งาน API

### Health Check

```bash
curl http://localhost:8000/health
```

### Rerank เอกสาร

```bash
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "documents": ["ML is a subset of AI.", "The weather is sunny."],
    "top_n": 2,
    "return_documents": true
  }'
```

### ตัวอย่างผลลัพธ์

```json
{
  "model": "jinaai/jina-reranker-v3",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.95,
      "document": {"text": "ML is a subset of AI."}
    },
    {
      "index": 1,
      "relevance_score": 0.12,
      "document": {"text": "The weather is sunny."}
    }
  ]
}
```

ดูเอกสาร API ฉบับเต็มได้ที่ [API Documentation](context/docs/api.md)

---

## Development

```bash
# Run tests
pytest tests/

# Lint
ruff check app/ tests/

# Type check
mypy app/