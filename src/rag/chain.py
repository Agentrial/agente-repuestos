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

# ── 5. Historial en memoria ──────────────────────────────────────────

historial: list = []

# ── 6. Función principal ─────────────────────────────────────────────

def consultar(pregunta: str) -> str:
    # Buscar contexto en ChromaDB
    docs = retriever.invoke(pregunta)
    contexto = "\n".join([f"- {doc.page_content}" for doc in docs])

    # Llamar al LLM con historial
    respuesta = chain.invoke({
        "context":   contexto,
        "historial": historial,
        "question":  pregunta,
    })

    # Guardar en historial
    historial.append(HumanMessage(content=pregunta))
    historial.append(AIMessage(content=respuesta))

    return respuesta