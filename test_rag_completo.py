"""
test_rag_completo.py
--------------------
Prueba el pipeline RAG completo:
consulta → ChromaDB → Gemini → respuesta en lenguaje natural
"""

import chromadb
from sentence_transformers import SentenceTransformer
from src.llm.gemini_client import consultar

# ── 1. Conectar a ChromaDB ───────────────────────────────────────────

modelo    = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
cliente   = chromadb.PersistentClient(path="data/chromadb")
coleccion = cliente.get_collection("repuestos")

# ── 2. Buscar repuestos relevantes ───────────────────────────────────

pregunta = "el tractor pierde aceite por el motor"

vector = modelo.encode(pregunta).tolist()

resultados = coleccion.query(
    query_embeddings=[vector],
    n_results=5,
)

repuestos = [
    {
        "descripcion":   meta["descripcion"],
        "numero_parte":  meta["numero_parte"],
        "subsistema":    meta["subsistema"],
        "cantidad":      meta["cantidad"],
        "similitud":     round(1 - dist, 3),
    }
    for meta, dist in zip(
        resultados["metadatas"][0],
        resultados["distances"][0],
    )
]

print(f"Consulta: '{pregunta}'")
print(f"\nRepuestos encontrados por ChromaDB:")
for r in repuestos:
    print(f"  {r['similitud']} → {r['descripcion']} (#{r['numero_parte']})")

# ── 3. Generar respuesta con Gemini ──────────────────────────────────

print("\nGenerando respuesta con Gemini...\n")
print("=" * 60)
respuesta = consultar(pregunta, repuestos)
print(respuesta)
print("=" * 60)
