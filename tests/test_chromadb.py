"""
tests/test_chromadb.py
----------------------
Verifica que ChromaDB devuelve resultados relevantes.
"""

import pytest
import chromadb
from sentence_transformers import SentenceTransformer

@pytest.fixture(scope="module")
def coleccion():
    cliente = chromadb.PersistentClient(path="data/chromadb")
    return cliente.get_collection("repuestos")

@pytest.fixture(scope="module")
def modelo():
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")


def test_coleccion_tiene_repuestos(coleccion):
    assert coleccion.count() > 0, "La colección está vacía"


def test_coleccion_tiene_cantidad_esperada(coleccion):
    assert coleccion.count() == 602, f"Se esperaban 602 repuestos, hay {coleccion.count()}"


def test_busqueda_devuelve_resultados_relevantes(coleccion, modelo):
    vector = modelo.encode("problema con las válvulas del motor").tolist()
    resultados = coleccion.query(query_embeddings=[vector], n_results=3)
    documentos = resultados["documents"][0]
    assert len(documentos) == 3
    # Al menos uno debe mencionar válvula
    assert any("álvula" in doc.lower() for doc in documentos), \
        "Ningún resultado menciona válvula"


def test_similitud_minima(coleccion, modelo):
    vector = modelo.encode("cárter de aceite motor").tolist()
    resultados = coleccion.query(query_embeddings=[vector], n_results=1)
    distancia = resultados["distances"][0][0]
    similitud = 1 - distancia
    assert similitud > 0.70, f"Similitud muy baja: {similitud:.3f}"