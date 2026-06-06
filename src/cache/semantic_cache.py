"""
src/cache/semantic_cache.py
---------------------------
Caché semántico sobre ChromaDB con:
  - TTL por entrada (lazy expiration)
  - Hit/miss counters para métricas
  - Métodos de invalidación por query y total
"""
import time
import uuid
import chromadb


class SemanticCache:
    def __init__(self, path: str = "data/cache", embeddings=None, ttl_segundos: int = 3600, umbral_similitud: float = 0.85):
        self._embeddings = embeddings
        self._ttl = ttl_segundos
        self._umbral = umbral_similitud

        # Contadores en memoria — se resetean al reiniciar la app
        self._hits = 0
        self._misses = 0

        self.cliente = chromadb.PersistentClient(path=path)
        self.coleccion = self.cliente.get_or_create_collection(
            name="respuestas_cache",
            metadata={"hnsw:space": "cosine"},
        )

    def _encode(self, texto: str) -> list:
        return self._embeddings.embed_query(texto)

    def buscar(self, consulta: str) -> str | None:
        """
        Busca una respuesta cacheada para la consulta dada.
        Respeta TTL: entradas expiradas se ignoran y eliminan (lazy expiration).
        Retorna la respuesta si hay hit, None si hay miss.
        """
        if self.coleccion.count() == 0:
            self._misses += 1
            return None

        vector = self._encode(consulta)
        resultados = self.coleccion.query(
            query_embeddings=[vector],
            n_results=1,
        )

        distancia = resultados["distances"][0][0]
        similitud = 1 - distancia

        if similitud >= self._umbral:
            metadata = resultados["metadatas"][0][0]
            doc_id = resultados["ids"][0][0]

            # Verificar TTL
            timestamp = metadata.get("timestamp", 0)
            edad_segundos = time.time() - timestamp
            if edad_segundos > self._ttl:
                # Entrada expirada — borrar y reportar miss
                self.coleccion.delete(ids=[doc_id])
                print(f"[CACHÉ EXPIRED] similitud: {similitud:.3f}, edad: {edad_segundos:.0f}s")
                self._misses += 1
                return None

            print(f"[CACHÉ HIT] similitud: {similitud:.3f}, edad: {edad_segundos:.0f}s")
            self._hits += 1
            return metadata["respuesta"]

        print(f"[CACHÉ MISS] similitud máxima: {similitud:.3f}")
        self._misses += 1
        return None

    def guardar(self, consulta: str, respuesta: str) -> None:
        """
        Guarda una nueva entrada en caché con timestamp actual.
        """
        vector = self._encode(consulta)
        cache_id = f"cache_{uuid.uuid4().hex}"
        self.coleccion.add(
            ids=[cache_id],
            embeddings=[vector],
            documents=[consulta],
            metadatas=[{
                "respuesta": respuesta,
                "timestamp": time.time(),   # Para TTL
            }],
        )
        print(f"[CACHÉ SAVE] id: {cache_id}")

    def invalidar(self, consulta: str) -> bool:
        if self.coleccion.count() == 0:  
            return False 
        """
        Invalida la entrada más similar a la consulta dada (si supera umbral).
        Retorna True si se eliminó algo, False si no había nada que invalidar.
        """
        if self.coleccion.count() == 0:
            return False

        vector = self._encode(consulta)
        resultados = self.coleccion.query(
            query_embeddings=[vector],
            n_results=1,
        )

        distancia = resultados["distances"][0][0]
        similitud = 1 - distancia

        if similitud >= self._umbral:
            doc_id = resultados["ids"][0][0]
            self.coleccion.delete(ids=[doc_id])
            print(f"[CACHÉ INVALIDATE] id: {doc_id}, similitud: {similitud:.3f}")
            return True

        print(f"[CACHÉ INVALIDATE] Nada que invalidar, similitud máxima: {similitud:.3f}")
        return False

    def limpiar_todo(self) -> int:
        """
        Elimina todas las entradas del caché.
        Retorna el número de entradas eliminadas.
        """
        total = self.coleccion.count()
        nombre = self.coleccion.name
        # Recrear la colección es más limpio que borrar por IDs
        self.cliente.delete_collection(nombre)
        self.coleccion = self.cliente.get_or_create_collection(
            name=nombre,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[CACHÉ CLEAR] Eliminadas {total} entradas")
        return total

    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total > 0 else 0.0
        return {
            "entradas_en_cache": self.coleccion.count(),
            "umbral_similitud": self._umbral,
            "ttl_segundos": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
        }