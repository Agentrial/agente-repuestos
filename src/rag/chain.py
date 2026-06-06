"""
src/rag/chain.py
----------------
Pipeline RAG con:
  - Búsqueda híbrida: BM25 + semántica fusionada con Reciprocal Rank Fusion
  - Historial stateless (el cliente envía su propio historial)
  - Sliding window para limitar el historial al LLM
  - asyncio.to_thread para no bloquear el event loop
  - Timeout configurable en la llamada al LLM
"""
import os
import asyncio
import yaml
from pathlib import Path
from dotenv import load_dotenv
 
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
 
from src.cache.semantic_cache import SemanticCache
 
load_dotenv()
 
# ── Estado lazy (inicializado una sola vez en lifespan) ──────────────
_initialized = False
_retriever   = None
_bm25        = None          # Índice BM25 en memoria
_bm25_docs   = []            # Lista de Document — mismo orden que el índice
_chain       = None
_cache       = None
_config      = None          # Config completa del YAML
 
 
# ── RRF ─────────────────────────────────────────────────────────────
 
def _reciprocal_rank_fusion(
    lista_a: list[Document],
    lista_b: list[Document],
    k: int = 60,
    weight_a: float = 0.7,
    weight_b: float = 0.3,
) -> list[Document]:
    """
    Combina dos listas de documentos ordenadas por relevancia usando
    Reciprocal Rank Fusion.
 
    score(doc) = weight_a * 1/(k + rank_en_a) + weight_b * 1/(k + rank_en_b)
 
    Los documentos que aparecen solo en una lista reciben score 0 de la otra.
    Retorna lista ordenada descendente por score fusionado.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
 
    for rank, doc in enumerate(lista_a):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + weight_a * (1.0 / (k + rank))
        doc_map[key] = doc
 
    for rank, doc in enumerate(lista_b):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + weight_b * (1.0 / (k + rank))
        doc_map[key] = doc
 
    ordenados = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in ordenados]
 
 
# ── Inicialización ───────────────────────────────────────────────────
 
def _init():
    """Inicializa todos los componentes pesados una sola vez."""
    global _initialized, _retriever, _bm25, _bm25_docs, _chain, _cache, _config
 
    if _initialized:
        return
 
    # 0. Cargar configuración
    ruta_config = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"
    with ruta_config.open(encoding="utf-8") as f:
        _config = yaml.safe_load(f)
 
    prompt_cfg    = _config["system_prompt"]
    model_cfg     = _config["model"]
    cache_cfg     = _config["cache"]
    retrieval_cfg = _config["retrieval"]
 
    system_prompt = f"""{prompt_cfg['rol']}
    Idioma: {prompt_cfg['idioma']}
    Tono: {prompt_cfg['tono']}
 
    Usá únicamente los siguientes repuestos del catálogo para responder:
    {{context}}
 
    {prompt_cfg['formato_respuesta']}
 
    Restricciones:
    {chr(10).join(f'- {r}' for r in prompt_cfg['restricciones'])}"""
 
    # 1. Descargar ChromaDB si no existe
    if not Path("data/chromadb").exists():
        print("Descargando ChromaDB desde HuggingFace Hub...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="angeldeveloper256/cotizador-chromadb",
            repo_type="dataset",
            local_dir="data/chromadb",
            token=os.environ.get("HF_TOKEN"),
        )
        print("ChromaDB descargado.")
 
    # 2. Embeddings (instancia única compartida con el caché)
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-mpnet-base-v2"
    )
 
    # 3. Vectorstore + retriever semántico
    vectorstore = Chroma(
        persist_directory="data/chromadb",
        embedding_function=embeddings,
        collection_name="repuestos",
    )
    _retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": retrieval_cfg["n_results"]},
    )
 
    # 4. Índice BM25 — construido sobre todos los documentos de ChromaDB
    #    get() sin filtros devuelve todos los registros con sus textos
    print("Construyendo índice BM25...")
    print(f"ChromaDB path: data/chromadb")
    print(f"Colecciones disponibles: {[c.name for c in vectorstore._client.list_collections()]}")
    raw = vectorstore.get(include=["documents", "metadatas"])
    print(f"Documentos recuperados por get(): {len(raw['documents'])}")
    for root, dirs, files in os.walk("data/chromadb"):
        for file in files:
            fpath = os.path.join(root, file)
            print(f"  FILE: {fpath} ({os.path.getsize(fpath)} bytes)")
    _bm25_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    # Tokenización simple: lowercase + split por espacios
    # Para números de parte como "RE504836" esto es suficiente
    corpus_tokenizado = [doc.page_content.lower().split() for doc in _bm25_docs if doc.page_content.strip()]
    print(f"Documentos para BM25: {len(corpus_tokenizado)}")
    if corpus_tokenizado:
        _bm25 = BM25Okapi(corpus_tokenizado)
        print(f"Índice BM25 construido con {len(_bm25_docs)} documentos.")
    else:
        _bm25 = None
        print("WARN: corpus vacío, BM25 deshabilitado — solo búsqueda semántica.")
 
    # 5. LLM
    llm = ChatGoogleGenerativeAI(
        model=model_cfg["nombre"],
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=model_cfg["temperature"],
        max_output_tokens=model_cfg["max_output_tokens"],
    )
 
    # 6. Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="historial"),
        ("human", "{question}"),
    ])
    _chain = prompt | llm | StrOutputParser()
 
    # 7. Caché semántico — reutiliza la misma instancia de embeddings
    _cache = SemanticCache(
        embeddings=embeddings,
        ttl_segundos=cache_cfg["ttl_segundos"],
        umbral_similitud=cache_cfg["umbral_similitud"],
    )
 
    _initialized = True
    print("Pipeline RAG inicializado.")
 
 
# ── Búsqueda híbrida ─────────────────────────────────────────────────
 
def _buscar_hibrido(pregunta: str, n_results: int) -> list[Document]:
    """
    Combina búsqueda semántica (ChromaDB) con keyword (BM25) usando RRF.
    Retorna los top n_results documentos fusionados.
    """
    retrieval_cfg = _config["retrieval"]
 
    # Búsqueda semántica
    docs_chroma = _retriever.invoke(pregunta)
 
    # Búsqueda BM25 — tokens de la pregunta
    if _bm25 is not None:
        tokens = pregunta.lower().split()
        scores_bm25 = _bm25.get_scores(tokens)
        n_candidatos = n_results * 2
        indices_top = sorted(
            range(len(scores_bm25)),
            key=lambda i: scores_bm25[i],
            reverse=True
        )[:n_candidatos]
        docs_bm25 = [_bm25_docs[i] for i in indices_top if scores_bm25[i] > 0]
    else:
        docs_bm25 = []
 
    # Fusión RRF
    docs_fusionados = _reciprocal_rank_fusion(
        lista_a=docs_chroma,
        lista_b=docs_bm25,
        k=retrieval_cfg["rrf_k"],
        weight_a=1.0 - retrieval_cfg["bm25_weight"],
        weight_b=retrieval_cfg["bm25_weight"],
    )
 
    return docs_fusionados[:n_results]
 
 
# ── Función principal (async) ────────────────────────────────────────
 
async def consultar(
    pregunta: str,
    historial: list[dict],
) -> tuple[str, bool]:
    """
    Ejecuta el pipeline RAG completo.
 
    Args:
        pregunta:  Texto de la consulta del usuario.
        historial: Lista de mensajes previos enviada por el cliente.
                   Formato: [{"role": "user"|"assistant", "content": "..."}]
 
    Returns:
        (respuesta, desde_cache)
    """
    _init()
 
    # 1. Consultar caché semántico
    respuesta_cacheada = _cache.buscar(pregunta)
    if respuesta_cacheada:
        return respuesta_cacheada, True
 
    # 2. Convertir historial del cliente a mensajes LangChain
    mensajes_lc = []
    for msg in historial:
        if msg["role"] == "user":
            mensajes_lc.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            mensajes_lc.append(AIMessage(content=msg["content"]))
 
    # 3. Sliding window — limitar a los últimos max_turnos turnos
    #    Cada turno = 1 human + 1 AI = 2 mensajes
    max_mensajes = _config["historial"]["max_turnos"] * 2
    historial_ventana = mensajes_lc[-max_mensajes:] if len(mensajes_lc) > max_mensajes else mensajes_lc
 
    # 4. Búsqueda híbrida
    n_results = _config["retrieval"]["n_results"]
    docs = _buscar_hibrido(pregunta, n_results)
    contexto = "\n".join([f"- {doc.page_content}" for doc in docs])
 
    # 5. Llamada al LLM — síncrona dentro de asyncio.to_thread
    #    asyncio.to_thread libera el event loop mientras Gemini responde
    #    asyncio.wait_for aplica el timeout configurado
    timeout = _config["model"]["timeout_segundos"]
 
    def _invocar_chain():
        return _chain.invoke({
            "context":   contexto,
            "historial": historial_ventana,
            "question":  pregunta,
        })
 
    respuesta = await asyncio.wait_for(
        asyncio.to_thread(_invocar_chain),
        timeout=timeout,
    )
 
    # 6. Guardar en caché
    _cache.guardar(pregunta, respuesta)
 
    return respuesta, False