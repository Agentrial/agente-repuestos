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
import yaml

load_dotenv()

# ── Estado lazy ──────────────────────────────────────────────────────
_initialized = False
_retriever = None
_chain = None
_cache = None
_historiales: dict = {}

def _init():
    """Inicializa todos los componentes pesados una sola vez."""
    global _initialized, _retriever, _chain, _cache

    if _initialized:
        return

    # 0. Cargar configuración
    with open("config/prompts.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    prompt_config = config["system_prompt"]
    model_config = config["model"]

    system_prompt = f"""{prompt_config['rol']}
    Idioma: {prompt_config['idioma']}
    Tono: {prompt_config['tono']}

    Usá únicamente los siguientes repuestos del catálogo para responder:
    {{context}}

    {prompt_config['formato_respuesta']}

    Restricciones:
    {chr(10).join(f"- {r}" for r in prompt_config['restricciones'])}"""

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
    model=model_config["nombre"],
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=model_config["temperature"],
    )
    # 4. Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="historial"),
        ("human", "{question}"),

    ])
    _chain = prompt | llm | StrOutputParser()

    # 5. Caché — reutiliza la misma instancia de embeddings
    _cache = SemanticCache(embeddings=embeddings)

    _initialized = True


# ── Función principal ────────────────────────────────────────────────
def consultar(pregunta: str, session_id: str = "default") -> tuple[str, bool]:
    _init()

    respuesta_cacheada = _cache.buscar(pregunta)
    if respuesta_cacheada:
        return respuesta_cacheada, True

    # Obtener o crear historial para esta sesión
    if session_id not in _historiales:
        _historiales[session_id] = []
    historial = _historiales[session_id]

    docs = _retriever.invoke(pregunta)
    contexto = "\n".join([f"- {doc.page_content}" for doc in docs])

    respuesta = _chain.invoke({
        "context":   contexto,
        "historial": historial,
        "question":  pregunta,
    })

    _cache.guardar(pregunta, respuesta)
    historial.append(HumanMessage(content=pregunta))
    historial.append(AIMessage(content=respuesta))

    return respuesta, False