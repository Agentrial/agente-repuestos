"""
src/api/main.py
---------------
API REST para el agente de repuestos John Deere.
 
Mejoras incluidas:
  - Historial stateless: el cliente envía su propio historial
  - Rate limiting con slowapi (por IP)
  - API Key authentication via header X-API-Key
  - Manejo de excepciones: 503 en timeout/error de Gemini, 500 en error inesperado
  - POST /cache/invalidate para invalidar entradas específicas o todo el caché
  - GET /cache/stats con hit rate counter
"""
import asyncio
import logging
from contextlib import asynccontextmanager
 
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
 
from src.rag.chain import consultar as rag_consultar, _init
from src.rag.chain import _cache as rag_cache
 
import os
from dotenv import load_dotenv
 
load_dotenv()
 
logger = logging.getLogger(__name__)
 
# ── Rate limiter ─────────────────────────────────────────────────────
# key_func=get_remote_address → el límite se aplica por IP del cliente
limiter = Limiter(key_func=get_remote_address)
 
 
# ── Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _init()
    yield
 
 
# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agente Repuestos — John Deere 5090E",
    description="API RAG para consultas de repuestos usando HuggingFace + ChromaDB + Gemini",
    version="2.0.0",
    lifespan=lifespan,
)
 
# Registrar el handler de 429 que slowapi espera encontrar en el app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
 
 
# ── API Key authentication ────────────────────────────────────────────
# Las keys válidas se definen en .env como: API_KEYS=key1,key2,key3
# Si API_KEYS no está definida, la autenticación está deshabilitada
# (útil para desarrollo local)
 
def _cargar_api_keys() -> set[str]:
    raw = os.getenv("API_KEYS", "")
    if not raw.strip():
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}
 
_API_KEYS = _cargar_api_keys()
 
def verificar_api_key(x_api_key: str | None = Header(default=None)) -> str | None:
    """
    Dependency de FastAPI para autenticación por API Key.
 
    - Si API_KEYS no está configurado en .env → autenticación deshabilitada,
      permite todos los requests (modo desarrollo).
    - Si API_KEYS está configurado:
        - Header ausente → 401 Unauthorized
        - Header presente pero key inválida → 403 Forbidden
        - Header presente y key válida → continúa
    """
    if not _API_KEYS:
        # Autenticación deshabilitada — modo desarrollo
        return None
 
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Header X-API-Key requerido",
        )
 
    if x_api_key not in _API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="API Key inválida",
        )
 
    return x_api_key
 
 
# ── Modelos de datos ─────────────────────────────────────────────────
 
class MensajeHistorial(BaseModel):
    role: str       # "user" o "assistant"
    content: str
 
class ConsultaRequest(BaseModel):
    pregunta: str
    historial: list[MensajeHistorial] = []  # El cliente envía su propio historial
 
class ConsultaResponse(BaseModel):
    respuesta: str
    desde_cache: bool
 
class InvalidarCacheRequest(BaseModel):
    query: str | None = None    # Si es None, limpia todo el caché
 
 
# ── Endpoints ────────────────────────────────────────────────────────
 
@app.get("/")
def root():
    return {
        "servicio": "Agente Repuestos John Deere 5090E",
        "version": "2.0.0",
        "estado": "activo",
    }
 
 
@app.get("/health")
def health():
    return {"estado": "ok"}
 
 
@app.post("/consultar", response_model=ConsultaResponse)
@limiter.limit("20/minute")
async def consultar_repuesto(
    request: Request,                                      # Requerido por slowapi
    body: ConsultaRequest,
    _api_key: str | None = Depends(verificar_api_key),
):
    """
    Endpoint principal del agente RAG.
 
    El cliente es responsable de mantener y enviar su propio historial.
    El servidor no guarda estado de sesión.
 
    Errores:
      400 — Pregunta vacía
      401 — Header X-API-Key ausente (si autenticación está habilitada)
      403 — API Key inválida
      429 — Rate limit excedido
      503 — Gemini no disponible (timeout o error de la API)
      500 — Error interno inesperado
    """
    if not body.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")
 
    historial_dicts = [{"role": m.role, "content": m.content} for m in body.historial]
 
    try:
        respuesta, desde_cache = await rag_consultar(
            pregunta=body.pregunta,
            historial=historial_dicts,
        )
        return ConsultaResponse(respuesta=respuesta, desde_cache=desde_cache)
 
    except asyncio.TimeoutError:
        logger.error("Timeout esperando respuesta de Gemini")
        raise HTTPException(
            status_code=503,
            detail="El servicio de IA no respondió a tiempo. Intenta de nuevo en unos segundos.",
        )
 
    except Exception as exc:
        logger.exception("Error inesperado en /consultar: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor",
        )
 
 
@app.get("/cache/stats")
def cache_stats(_api_key: str | None = Depends(verificar_api_key)):
    """
    Retorna métricas del caché semántico.
    Incluye hit rate calculado desde el último reinicio de la app.
    """
    # Importar _cache aquí para obtener la referencia actualizada post-_init()
    from src.rag.chain import _cache
    if _cache is None:
        return {"entradas_en_cache": 0, "hits": 0, "misses": 0, "hit_rate": 0.0}
    return _cache.stats()
 
 
@app.post("/cache/invalidate")
def invalidar_cache(
    body: InvalidarCacheRequest,
    _api_key: str | None = Depends(verificar_api_key),
):
    """
    Invalida entradas del caché semántico.
 
    - Con `query`: invalida la entrada más similar a esa query (si supera umbral).
    - Sin `query` (o query=null): limpia todo el caché.
 
    Ejemplos de body:
      {"query": "filtro de aceite RE504836"}  → invalida entrada específica
      {}                                       → limpia todo el caché
    """
    from src.rag.chain import _cache
    if _cache is None:
        raise HTTPException(status_code=503, detail="El pipeline RAG no está inicializado")
 
    if body.query:
        eliminado = _cache.invalidar(body.query)
        return {
            "invalidado": eliminado,
            "mensaje": "Entrada invalidada" if eliminado else "No se encontró entrada similar",
        }
    else:
        total_eliminadas = _cache.limpiar_todo()
        return {
            "invalidado": True,
            "entradas_eliminadas": total_eliminadas,
            "mensaje": f"Caché limpiado: {total_eliminadas} entradas eliminadas",
        }