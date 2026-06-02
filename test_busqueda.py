import chromadb
from sentence_transformers import SentenceTransformer

modelo    = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
cliente   = chromadb.PersistentClient(path="data/chromadb")
coleccion = cliente.get_collection("repuestos")

consulta = "problema con las válvulas del motor"
vector   = modelo.encode(consulta).tolist()

resultados = coleccion.query(
    query_embeddings=[vector],
    n_results=5,
)

print(f"\nConsulta: '{consulta}'\n")
print("=" * 60)

for i, (doc, meta, distancia) in enumerate(zip(
    resultados["documents"][0],
    resultados["metadatas"][0],
    resultados["distances"][0],
), start=1):
    similitud = 1 - distancia
    print(f"\nResultado #{i}  —  similitud: {similitud:.3f}")
    print(f"  Descripción:  {meta['descripcion']}")
    print(f"  Número parte: {meta['numero_parte']}")
    print(f"  Sistema:      {meta['sistema']}")
    print(f"  Subsistema:   {meta['subsistema']}")
    print(f"  Cantidad:     {meta['cantidad']}")
    print(f"  Texto usado para vectorizar: {doc}")
    print("-" * 60)