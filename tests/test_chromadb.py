"""
tests/test_chromadb.py
----------------------
Verifica que ChromaDB funciona correctamente.
Usa datos de prueba en memoria — no depende de datos reales.
"""

import pytest
import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

REPUESTOS_PRUEBA = [
    "Válvula de admisión - Motor - Culata con válvulas",
    "Tapa de válvula - Motor - Tapa de válvulas",
    "Cárter de aceite - Motor - Cárter de aceite",
    "Motor de arranque - Motor - Sistema de arranque",
    "Filtro de aceite - Motor - Sistema de lubricación",
]


@pytest.fixture(scope="module")
def modelo():
    return SentenceTransformer(MODEL_NAME)


@pytest.fixture(scope="module")
def coleccion(modelo):
    """Crea una colección ChromaDB en memoria con datos de prueba."""
    cliente = chromadb.EphemeralClient()
    col = cliente.create_collection(
        name="repuestos_test",
        metadata={"hnsw:space": "cosine"},
    )
    vectores = modelo.encode(REPUESTOS_PRUEBA).tolist()
    col.add(
        ids=[f"test_{i}" for i in range(len(REPUESTOS_PRUEBA))],
        embeddings=vectores,
        documents=REPUESTOS_PRUEBA,
        metadatas=[{"descripcion": r} for r in REPUESTOS_PRUEBA],
    )
    return col


def test_coleccion_tiene_repuestos(coleccion):
    assert coleccion.count() > 0


def test_coleccion_tiene_cantidad_esperada(coleccion):
    assert coleccion.count() == len(REPUESTOS_PRUEBA)


def test_busqueda_devuelve_resultados_relevantes(coleccion, modelo):
    vector = modelo.encode("problema con las válvulas del motor").tolist()
    resultados = coleccion.query(query_embeddings=[vector], n_results=3)
    documentos = resultados["documents"][0]
    assert len(documentos) == 3
    assert any("álvula" in doc.lower() for doc in documentos)


def test_similitud_minima(coleccion, modelo):
    vector = modelo.encode("cárter de aceite motor").tolist()
    resultados = coleccion.query(query_embeddings=[vector], n_results=1)
    distancia = resultados["distances"][0][0]
    similitud = 1 - distancia
    assert similitud > 0.70, f"Similitud muy baja: {similitud:.3f}"