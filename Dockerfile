FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.4.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')"

COPY scripts/download_chromadb.py ./scripts/download_chromadb.py

ARG HF_TOKEN
RUN pip install --no-cache-dir "huggingface_hub[hf_xet]" && \
    HF_TOKEN=${HF_TOKEN} python scripts/download_chromadb.py

COPY src/ ./src/
COPY config/ ./config/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
