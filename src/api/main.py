"""
src/api/main.py
---------------
API REST para el agente-repuestos de repuestos John Deere.
Model serving con FastAPI.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag.chain import consultar
from src.cache.semantic_cache import SemanticCache

# ── Inicializar app ──────────────────────────────────────────────────

app = FastAPI(
    title="Agente Repuestos — John Deere 5090E",
    description="API RAG para consultas de repuestos usando HuggingFace + ChromaDB + Gemini",
    version="1.0.0",
)



# ── Modelos de datos ─────────────────────────────────────────────────

class ConsultaRequest(BaseModel):
    pregunta: str
    session_id: str = "default"

class ConsultaResponse(BaseModel):
    respuesta: str
    desde_cache: bool
    session_id: str

# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "servicio": "Agente Repuestos John Deere 5090E",
        "version": "1.0.0",
        "estado": "activo",
    }

@app.get("/health")
def health():
    return {"estado": "ok"}

@app.post("/consultar", response_model=ConsultaResponse)
def consultar_repuesto(request: ConsultaRequest):
    if not request.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")

    respuesta, desde_cache = consultar(request.pregunta, request.session_id)

    return ConsultaResponse(
        respuesta=respuesta,
        desde_cache=desde_cache,
        session_id=request.session_id,
    )

from src.rag.chain import _cache as rag_cache

@app.get("/cache/stats")
def cache_stats():
    if rag_cache is None:
        return {"entradas_en_cache": 0, "umbral_similitud": 0.85}
    return rag_cache.stats()