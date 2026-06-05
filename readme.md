# 🚜 Agente Repuestos — John Deere 5090E
 
![CI](https://github.com/Agentrial/agente-repuestos/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![HuggingFace](https://img.shields.io/badge/🤗_Spaces-running-green)
 
Agente RAG que reconoce un problema mecánico y devuelve el número de pieza necesario.
Diseñado para distribuidores de repuestos John Deere en Perú, con soporte para
coloquialismos del español peruano ("corazón del tractor" → Motor Completo #RE557577).
 
🔗 **Demo en producción:** https://angeldeveloper256-agente-repuestos.hf.space/docs
 
---
 
## Stack
 
| Capa | Tecnología |
|---|---|
| Embeddings | HuggingFace `paraphrase-multilingual-mpnet-base-v2` |
| Vector Store | ChromaDB (similitud coseno) |
| LLM | Google Gemini 2.5 Flash Lite (swappable) |
| RAG Pipeline | LangChain |
| Experimentos | MLflow |
| API | FastAPI |
| CI/CD | GitHub Actions + pytest |
| Deploy | HuggingFace Spaces (Docker) |
 
---
 
## Arquitectura
 
```
Consulta usuario
      ↓
Caché semántico (ChromaDB) ── hit ──→ Respuesta inmediata
      ↓ miss
Embeddings (HuggingFace)
      ↓
Retrieval top-3 (ChromaDB)
      ↓
Prompt + contexto + historial
      ↓
LLM (Gemini)
      ↓
Respuesta + guardar en caché
```
 
**Decisiones de diseño:**
- **Una sola instancia del modelo** compartida entre retriever y caché semántico — reduce consumo de RAM de ~840MB a ~420MB
- **Lazy loading** — el modelo se carga en la primera consulta, no al arrancar el servidor
- **Dataset privado en HuggingFace Hub** — los binarios de ChromaDB se descargan en runtime usando un token de mínimo privilegio
- **Provider-agnostic** — swappear el LLM requiere cambiar una línea en `chain.py`
---
 
## Uso
 
### Endpoint principal
 
```bash
curl -X POST https://angeldeveloper256-agente-repuestos.hf.space/consultar \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "necesito el corazón del tractor"}'
```
 
### Respuesta esperada
 
```json
{
  "respuesta": "DIAGNÓSTICO: El cliente necesita el motor del tractor.\nREPUESTOS:\n  1. Motor Completo #RE557577\nACCIÓN: Verificar disponibilidad del Motor Completo #RE557577.",
  "desde_cache": false,
  "session_id": "default"
}
```
 
### Documentación interactiva (Swagger)
 
```
https://angeldeveloper256-agente-repuestos.hf.space/docs
```
 
> **Nota:** El primer request puede tardar 30-60 segundos si el Space estuvo inactivo.
 
---
 
## Correr localmente
 
### Requisitos
- Python 3.11
- Docker
- API key de Google Gemini (gratuita en [aistudio.google.com](https://aistudio.google.com))
### Con Docker
 
```bash
docker pull angeldeveloper256/agente-repuestos:slim
 
docker run -p 7860:7860 \
  -e GEMINI_API_KEY=tu_api_key \
  -e HF_TOKEN=tu_hf_token \
  angeldeveloper256/agente-repuestos:slim
```
 
Luego abrí `http://localhost:7860/docs`
 
---
 
## Estructura
 
```
agente-repuestos/
├── src/
│   ├── api/
│   │   └── main.py           # FastAPI — endpoints
│   ├── rag/
│   │   └── chain.py          # Pipeline RAG con LangChain
│   └── cache/
│       └── semantic_cache.py # Caché semántico con ChromaDB
├── scripts/
│   └── load_catalog.py       # Vectoriza los 602 repuestos
├── experiments/
│   └── rag_experiment.py     # Experimentos MLflow
├── config/
│   └── prompts.yaml          # Configuración externalizada
├── tests/
│   └── ...                   # 7 tests con pytest
├── Dockerfile
└── .github/
    └── workflows/
        └── ci.yml            # GitHub Actions
```
 