"""
src/cache/semantic_cache.py
"""
import chromadb
import uuid

UMBRAL_SIMILITUD = 0.85

class SemanticCache:
    def __init__(self, path: str = "data/cache", embeddings=None):
        # Reutiliza el modelo pasado desde chain.py
        self._embeddings = embeddings
        self.cliente = chromadb.PersistentClient(path=path)
        self.coleccion = self.cliente.get_or_create_collection(
            name="respuestas_cache",
            metadata={"hnsw:space": "cosine"},
        )

    def _encode(self, texto: str) -> list:
        return self._embeddings.embed_query(texto)

    def buscar(self, consulta: str) -> str | None:
        if self.coleccion.count() == 0:
            return None
        vector = self._encode(consulta)
        resultados = self.coleccion.query(
            query_embeddings=[vector],
            n_results=1,
        )
        distancia = resultados["distances"][0][0]
        similitud = 1 - distancia
        if similitud >= UMBRAL_SIMILITUD:
            metadata = resultados["metadatas"][0][0]
            print(f"[CACHÉ HIT] similitud: {similitud:.3f}")
            return metadata["respuesta"]
        print(f"[CACHÉ MISS] similitud máxima: {similitud:.3f}")
        return None

    def guardar(self, consulta: str, respuesta: str) -> None:
        vector = self._encode(consulta)
        cache_id = f"cache_{uuid.uuid4().hex}"
        self.coleccion.add(
            ids=[cache_id],
            embeddings=[vector],
            documents=[consulta],
            metadatas=[{"respuesta": respuesta}],
        )
        print(f"[CACHÉ SAVE] id: {cache_id}")

    def stats(self) -> dict:
        return {
            "entradas_en_cache": self.coleccion.count(),
            "umbral_similitud":  UMBRAL_SIMILITUD,
        }