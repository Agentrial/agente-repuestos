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

# ── 1. Embeddings ────────────────────────────────────────────────────

embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-mpnet-base-v2"
)

# ── 2. Retriever ─────────────────────────────────────────────────────

vectorstore = Chroma(
    persist_directory="data/chromadb",
    embedding_function=embeddings,
    collection_name="repuestos",
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# ── 3. LLM ───────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
)

# ── 4. Prompt con historial ──────────────────────────────────────────

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

chain = prompt | llm | StrOutputParser()

# ── 5. Caché y historial ─────────────────────────────────────────────

cache = SemanticCache()
historial: list = []

# ── 6. Función principal ─────────────────────────────────────────────

def consultar(pregunta: str) -> tuple[str, bool]:
    """
    Ejecuta el pipeline RAG con caché semántico.
    Devuelve (respuesta, desde_cache).
    """
    # 1. Buscar en caché primero
    respuesta_cacheada = cache.buscar(pregunta)
    if respuesta_cacheada:
        return respuesta_cacheada, True

    # 2. Si no hay caché, buscar en ChromaDB
    docs = retriever.invoke(pregunta)
    contexto = "\n".join([f"- {doc.page_content}" for doc in docs])

    # 3. Llamar al LLM con historial
    respuesta = chain.invoke({
        "context":   contexto,
        "historial": historial,
        "question":  pregunta,
    })

    # 4. Guardar en caché y historial
    cache.guardar(pregunta, respuesta)
    historial.append(HumanMessage(content=pregunta))
    historial.append(AIMessage(content=respuesta))

    return respuesta, False