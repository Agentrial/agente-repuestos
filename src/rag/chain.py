"""
src/rag/chain.py
----------------
Pipeline RAG con memoria conversacional manual.
"""
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from src.cache.semantic_cache import SemanticCache

load_dotenv()

# ── Estado lazy ──────────────────────────────────────────────────────
_initialized = False
_retriever = None
_chain = None
_cache = None
_historial: list = []

def _init():
    """Inicializa todos los componentes pesados una sola vez."""
    global _initialized, _retriever, _chain, _cache

    if _initialized:
        return

    # 0. Descargar ChromaDB si no existe (runtime, usa HF_TOKEN del entorno)
    from pathlib import Path
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

    # 1. Embeddings (instancia única compartida)
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-mpnet-base-v2"
    )
    # ... resto igual

    # 2. Retriever
    vectorstore = Chroma(
        persist_directory="data/chromadb",
        embedding_function=embeddings,
        collection_name="repuestos",
    )
    _retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # 3. LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
    )

    # 4. Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un asistente técnico especializado en repuestos
para tractores John Deere 5090E.
Usá únicamente los siguientes repuestos del catálogo para responder:
{context}
Respondé en este formato:
DIAGNÓSTICO: [una línea]
REPUESTOS:
  1. [nombre] #[numero_parte]
ACCIÓN: [un paso concreto]"""),
        MessagesPlaceholder(variable_name="historial"),
        ("human", "{question}"),
    ])
    _chain = prompt | llm | StrOutputParser()

    # 5. Caché — reutiliza la misma instancia de embeddings
    _cache = SemanticCache(embeddings=embeddings)

    _initialized = True


# ── Función principal ────────────────────────────────────────────────
def consultar(pregunta: str) -> tuple[str, bool]:
    _init()

    respuesta_cacheada = _cache.buscar(pregunta)
    if respuesta_cacheada:
        return respuesta_cacheada, True

    docs = _retriever.invoke(pregunta)
    contexto = "\n".join([f"- {doc.page_content}" for doc in docs])

    respuesta = _chain.invoke({
        "context":   contexto,
        "historial": _historial,
        "question":  pregunta,
    })

    _cache.guardar(pregunta, respuesta)
    _historial.append(HumanMessage(content=pregunta))
    _historial.append(AIMessage(content=respuesta))

    return respuesta, False