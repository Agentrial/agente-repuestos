FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.4.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descargar modelo de embeddings durante el build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')"

# Descargar ChromaDB desde HuggingFace Hub
RUN pip install --no-cache-dir huggingface_hub[hf_xet] && \
    python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='angeldeveloper256/cotizador-chromadb', \
    repo_type='dataset', \
    local_dir='data/chromadb', \
    token='${HF_TOKEN}' \
)"

COPY src/ ./src/
COPY config/ ./config/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
