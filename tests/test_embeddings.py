"""
tests/test_embeddings.py
------------------------
Verifica que el modelo de embeddings funciona correctamente.
"""

import pytest
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

@pytest.fixture(scope="module")
def modelo():
    return SentenceTransformer(MODEL_NAME)


def test_vector_tiene_dimension_correcta(modelo):
    vector = modelo.encode("el motor pierde aceite")
    assert vector.shape == (768,)


def test_textos_similares_tienen_alta_similitud(modelo):
    from sklearn.metrics.pairwise import cosine_similarity
    v1 = modelo.encode("ruido al frenar")
    v2 = modelo.encode("chirría cuando freno")
    similitud = cosine_similarity([v1], [v2])[0][0]
    assert similitud > 0.70, f"Similitud esperada >0.70, obtenida: {similitud:.3f}"


def test_textos_distintos_tienen_baja_similitud(modelo):
    from sklearn.metrics.pairwise import cosine_similarity
    v1 = modelo.encode("ruido al frenar")
    v2 = modelo.encode("el motor no arranca")
    similitud = cosine_similarity([v1], [v2])[0][0]
    assert similitud < 0.70, f"Similitud esperada <0.70, obtenida: {similitud:.3f}"