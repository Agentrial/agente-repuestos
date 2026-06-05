"""
experiments/rag_experiment.py
------------------------------
Ejecuta el pipeline RAG completo registrando
parámetros y métricas en MLflow.
"""

import time
import mlflow
import chromadb
from sentence_transformers import SentenceTransformer
from src.llm.gemini_client import consultar, CONFIG

# ── Configuración del experimento ────────────────────────────────────

CONSULTAS_PRUEBA = [
    "el tractor pierde aceite por el motor",
    "el motor no arranca",
    "ruido extraño en el motor",
]

# ── Inicializar RAG ──────────────────────────────────────────────────

modelo    = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
cliente   = chromadb.PersistentClient(path="data/chromadb")
coleccion = cliente.get_collection("repuestos")

# ── Experimento MLflow ───────────────────────────────────────────────

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("agente-repuestos-rag")
model_cfg = CONFIG["model"]
N_RESULTADOS = 3

with mlflow.start_run(run_name="flash-lite-n3-temp01"):

    # Registrar parámetros de configuración
    mlflow.log_param("modelo_llm",        model_cfg["nombre"])
    mlflow.log_param("modelo_embeddings", "paraphrase-multilingual-mpnet-base-v2")
    mlflow.log_param("temperature",       model_cfg["temperature"])
    mlflow.log_param("n_resultados",      N_RESULTADOS)
    mlflow.log_param("thinking_budget",   0)

    tokens_totales   = []
    tokens_prompt    = []
    tokens_respuesta = []
    tiempos          = []
    completas        = 0

    for i, consulta in enumerate(CONSULTAS_PRUEBA, start=1):
        print(f"\nConsulta {i}/{len(CONSULTAS_PRUEBA)}: '{consulta}'")

        # Buscar en ChromaDB
        vector     = modelo.encode(consulta).tolist()
        resultados = coleccion.query(
            query_embeddings=[vector],
            n_results=N_RESULTADOS,
        )

        repuestos = [
            {
                "descripcion":  meta["descripcion"],
                "numero_parte": meta["numero_parte"],
                "subsistema":   meta["subsistema"],
                "cantidad":     meta["cantidad"],
                "similitud":    round(1 - dist, 3),
            }
            for meta, dist in zip(
                resultados["metadatas"][0],
                resultados["distances"][0],
            )
        ]

        # Llamar a Gemini y medir tiempo
        inicio    = time.time()
        respuesta, uso, finish = consultar(consulta, repuestos)
        duracion  = round(time.time() - inicio, 2)

        # Acumular métricas
        tokens_totales.append(uso.total_token_count)
        tokens_prompt.append(uso.prompt_token_count)
        tokens_respuesta.append(uso.candidates_token_count)
        tiempos.append(duracion)
        if str(finish) == "FinishReason.STOP":
            completas += 1

        # Guardar respuesta como artifact
        mlflow.log_text(respuesta, f"respuesta_{i}.txt")

        print(f"  tokens: {uso.total_token_count} | "
              f"tiempo: {duracion}s | finish: {finish}")

    # Registrar métricas promedio
    mlflow.log_metric("tokens_total_promedio",    sum(tokens_totales)   / len(tokens_totales))
    mlflow.log_metric("tokens_prompt_promedio",   sum(tokens_prompt)    / len(tokens_prompt))
    mlflow.log_metric("tokens_respuesta_promedio",sum(tokens_respuesta) / len(tokens_respuesta))
    mlflow.log_metric("tiempo_promedio_s",        sum(tiempos)          / len(tiempos))
    mlflow.log_metric("respuestas_completas",     completas)
    mlflow.log_metric("tasa_completitud",         completas / len(CONSULTAS_PRUEBA))

    print(f"\n✅ Experimento registrado en MLflow.")
    print(f"   Respuestas completas: {completas}/{len(CONSULTAS_PRUEBA)}")