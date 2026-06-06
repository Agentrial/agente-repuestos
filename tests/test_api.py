"""
tests/test_api.py
-----------------
Prueba el endpoint POST /consultar con el nuevo modelo stateless.
Mockea el pipeline RAG completo — no necesita Gemini ni ChromaDB.
"""

import asyncio
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock


@pytest.fixture(scope="module")
def client():
    with patch("src.rag.chain._init"):
        from src.api.main import app
        # Deshabilitar autenticación para los tests generales
        app.dependency_overrides = {}
        return TestClient(app)


def _post(client, pregunta: str, historial: list = None, api_key: str = None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    return client.post(
        "/consultar",
        json={"pregunta": pregunta, "historial": historial or []},
        headers=headers,
    )


# ── Tests con auth deshabilitada ──────────────────────────────────────
# Parcheamos _API_KEYS a set vacío para que verificar_api_key no bloquee

def test_consulta_basica_sin_historial(client):
    """Request mínimo válido debe devolver 200 con los campos esperados."""
    with patch("src.api.main._API_KEYS", set()):
        with patch("src.api.main.rag_consultar", new_callable=AsyncMock) as mock_rag:
            mock_rag.return_value = ("Respuesta de prueba", False)
            response = _post(client, "filtro de aceite")

    assert response.status_code == 200
    data = response.json()
    assert data["respuesta"] == "Respuesta de prueba"
    assert data["desde_cache"] is False


def test_consulta_con_historial(client):
    """El historial se pasa correctamente al pipeline RAG."""
    historial = [
        {"role": "user", "content": "Hola"},
        {"role": "assistant", "content": "Buen día"},
    ]
    with patch("src.api.main._API_KEYS", set()):
        with patch("src.api.main.rag_consultar", new_callable=AsyncMock) as mock_rag:
            mock_rag.return_value = ("Respuesta con contexto", False)
            response = _post(client, "¿y el filtro?", historial=historial)

    assert response.status_code == 200
    llamada = mock_rag.call_args
    assert len(llamada.kwargs["historial"]) == 2


def test_respuesta_desde_cache(client):
    """desde_cache: true se refleja correctamente en la respuesta."""
    with patch("src.api.main._API_KEYS", set()):
        with patch("src.api.main.rag_consultar", new_callable=AsyncMock) as mock_rag:
            mock_rag.return_value = ("Respuesta cacheada", True)
            response = _post(client, "filtro de aceite")

    assert response.status_code == 200
    assert response.json()["desde_cache"] is True


def test_pregunta_vacia_da_400(client):
    """Pregunta vacía debe devolver 400."""
    with patch("src.api.main._API_KEYS", set()):
        response = _post(client, "   ")
    assert response.status_code == 400


def test_timeout_gemini_da_503(client):
    """TimeoutError al llamar al LLM debe devolver 503."""
    with patch("src.api.main._API_KEYS", set()):
        with patch("src.api.main.rag_consultar", new_callable=AsyncMock) as mock_rag:
            mock_rag.side_effect = asyncio.TimeoutError()
            response = _post(client, "filtro de aceite")

    assert response.status_code == 503


def test_error_inesperado_da_500(client):
    """Excepción no manejada debe devolver 500."""
    with patch("src.api.main._API_KEYS", set()):
        with patch("src.api.main.rag_consultar", new_callable=AsyncMock) as mock_rag:
            mock_rag.side_effect = RuntimeError("Error inesperado")
            response = _post(client, "filtro de aceite")

    assert response.status_code == 500


# ── Tests de autenticación (con _API_KEYS activo) ─────────────────────

def test_api_key_ausente_da_401(client):
    """Sin header X-API-Key debe devolver 401 cuando auth está habilitada."""
    with patch("src.api.main._API_KEYS", {"mi-key-valida"}):
        response = _post(client, "filtro de aceite")
    assert response.status_code == 401


def test_api_key_invalida_da_403(client):
    """Key incorrecta debe devolver 403."""
    with patch("src.api.main._API_KEYS", {"mi-key-valida"}):
        response = _post(client, "filtro de aceite", api_key="key-incorrecta")
    assert response.status_code == 403


def test_api_key_valida_da_200(client):
    """Key correcta debe pasar la autenticación."""
    with patch("src.api.main._API_KEYS", {"mi-key-valida"}):
        with patch("src.api.main.rag_consultar", new_callable=AsyncMock) as mock_rag:
            mock_rag.return_value = ("OK", False)
            response = _post(client, "filtro de aceite", api_key="mi-key-valida")
    assert response.status_code == 200