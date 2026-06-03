FROM python:3.11-slim

WORKDIR /app

# 1. Instalar librerías del sistema necesarias para PyTorch en imágenes slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Instalar PyTorch CPU (Te sugiero la 2.3.1 por compatibilidad con tu sentence-transformers)
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir "numpy<2"

# 3. Resto de dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

EXPOSE 8000

# 4. Comando para Render
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}