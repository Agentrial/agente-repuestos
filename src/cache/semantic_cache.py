"""
src/cache/semantic_cache.py
---------------------------
Caché semántico para respuestas del LLM.
Evita llamar a Gemini cuando ya existe una respuesta
para una consulta semánticamente similar.
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer

UMBRAL_SIMILITUD = 0.85  # Si similitud > 0.85 → usar caché
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"


class SemanticCache:

    def __init__(self, path: str = "data/cache"):
        self.modelo = SentenceTransformer(MODEL_NAME)
        self.cliente = chromadb.PersistentClient(path=path)
        self.coleccion = self.cliente.get_or_create_collection(
            name="respuestas_cache",
            metadata={"hnsw:space": "cosine"},
        )

    def buscar(self, consulta: str) -> str | None:
        """
        Busca si existe una respuesta cacheada para una consulta similar.
        Devuelve la respuesta si la similitud supera el umbral, None si no.
        """
        if self.coleccion.count() == 0:
            return None

        vector = self.modelo.encode(consulta).tolist()
        resultados = self.coleccion.query(
            query_embeddings=[vector],
            n_results=1,
        )

        distancia  = resultados["distances"][0][0]
        similitud  = 1 - distancia

        if similitud >= UMBRAL_SIMILITUD:
            metadata = resultados["metadatas"][0][0]
            respuesta = metadata["respuesta"]
            print(f"[CACHÉ HIT] similitud: {similitud:.3f}")
            return respuesta

        print(f"[CACHÉ MISS] similitud máxima: {similitud:.3f}")
        return None

    def guardar(self, consulta: str, respuesta: str) -> None:
        """Guarda una consulta y su respuesta en el caché."""
        vector = self.modelo.encode(consulta).tolist()
        cache_id = f"cache_{self.coleccion.count()}"

        self.coleccion.add(
            ids=[cache_id],
            embeddings=[vector],
            documents=[consulta],
            metadatas=[{"respuesta": respuesta}],
        )
        print(f"[CACHÉ SAVE] consulta guardada con id: {cache_id}")

    def stats(self) -> dict:
        """Devuelve estadísticas del caché."""
        return {
            "entradas_en_cache": self.coleccion.count(),
            "umbral_similitud":  UMBRAL_SIMILITUD,
        }