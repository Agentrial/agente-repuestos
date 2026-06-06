"""
tests/test_cache_semantico.py
------------------------------
Prueba el caché semántico con fixtures en memoria.
No depende de ChromaDB real ni de Gemini.
"""

import time
import uuid
import pytest
import chromadb
from unittest.mock import MagicMock
from src.cache.semantic_cache import SemanticCache

# ── Vectores fijos para embeddings mockeados ─────────────────────────
VECTORES = {
    "filtro de aceite": [1.0, 0.0, 0.0],
    "filtro aceite motor": [0.95, 0.05, 0.0],
    "motor no arranca":   [0.0, 1.0, 0.0],
}

def _mock_embeddings(texto: str) -> list[float]:
    return VECTORES.get(texto, [0.5, 0.5, 0.0])


@pytest.fixture
def cache():
    """
    Instancia de SemanticCache completamente aislada por test.
    Cada test recibe una colección con nombre único — evita estado compartido
    entre instancias de EphemeralClient en el mismo proceso.
    """
    embeddings_mock = MagicMock()
    embeddings_mock.embed_query.side_effect = _mock_embeddings

    nombre_coleccion = f"cache_test_{uuid.uuid4().hex}"

    cliente = chromadb.EphemeralClient()
    coleccion = cliente.get_or_create_collection(
        name=nombre_coleccion,
        metadata={"hnsw:space": "cosine"},
    )

    instancia = SemanticCache(
        embeddings=embeddings_mock,
        ttl_segundos=2,
        umbral_similitud=0.85,
    )
    instancia.cliente = cliente
    instancia.coleccion = coleccion
    return instancia


# ── Tests ─────────────────────────────────────────────────────────────

def test_miss_en_cache_vacio(cache):
    """Cache vacío siempre devuelve None."""
    resultado = cache.buscar("filtro de aceite")
    assert resultado is None


def test_guardar_y_recuperar(cache):
    """Guardar una entrada y recuperarla debe dar hit."""
    cache.guardar("filtro de aceite", "Respuesta de prueba")
    resultado = cache.buscar("filtro de aceite")
    assert resultado == "Respuesta de prueba"


def test_consulta_identica_da_hit(cache):
    """La misma consulta exacta debe dar cache hit."""
    cache.guardar("filtro de aceite", "Respuesta A")
    resultado = cache.buscar("filtro de aceite")
    assert resultado is not None


def test_hit_rate_incrementa_correctamente(cache):
    """Hits y misses se cuentan correctamente."""
    # Cache vacío → miss
    cache.buscar("filtro de aceite")
    assert cache._misses == 1
    assert cache._hits == 0

    # Guardar y buscar → hit
    cache.guardar("filtro de aceite", "Respuesta A")
    cache.buscar("filtro de aceite")
    assert cache._hits == 1
    assert cache._misses == 1

    stats = cache.stats()
    assert stats["hit_rate"] == 0.5
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_ttl_expira_entrada(cache):
    """Una entrada con TTL expirado debe tratarse como miss y eliminarse."""
    cache.guardar("filtro de aceite", "Respuesta A")
    assert cache.buscar("filtro de aceite") is not None
    time.sleep(3)  # TTL del fixture es 2s
    resultado = cache.buscar("filtro de aceite")
    assert resultado is None


def test_ttl_no_expira_entrada_reciente(cache):
    """Una entrada reciente no debe expirar antes del TTL."""
    cache.guardar("filtro de aceite", "Respuesta A")
    time.sleep(1)  # Menos que el TTL de 2s
    resultado = cache.buscar("filtro de aceite")
    assert resultado is not None


def test_invalidar_entrada_existente(cache):
    """Invalidar una query existente debe devolver True y eliminarla."""
    # Colección limpia — solo una entrada
    cache.guardar("filtro de aceite", "Respuesta A")
    assert cache.coleccion.count() == 1

    eliminado = cache.invalidar("filtro de aceite")
    assert eliminado is True
    assert cache.coleccion.count() == 0
    assert cache.buscar("filtro de aceite") is None


def test_invalidar_entrada_inexistente(cache):
    """Invalidar en cache vacío debe devolver False."""
    # Colección vacía — count == 0, buscar retorna None antes del query
    assert cache.coleccion.count() == 0
    eliminado = cache.invalidar("filtro de aceite")
    assert eliminado is False


def test_limpiar_todo(cache):
    """limpiar_todo debe eliminar todas las entradas."""
    cache.guardar("filtro de aceite", "Respuesta A")
    cache.guardar("motor no arranca", "Respuesta B")
    assert cache.coleccion.count() == 2

    eliminadas = cache.limpiar_todo()
    assert eliminadas == 2
    assert cache.coleccion.count() == 0


def test_stats_estructura_completa(cache):
    """stats() debe devolver todos los campos esperados."""
    stats = cache.stats()
    assert "entradas_en_cache" in stats
    assert "umbral_similitud" in stats
    assert "ttl_segundos" in stats
    assert "hits" in stats
    assert "misses" in stats
    assert "hit_rate" in stats