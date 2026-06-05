"""
scripts/load_catalog.py
-----------------------
Carga los catálogos Excel, elimina duplicados y prepara
el texto para vectorizar en ChromaDB.
"""

import pandas as pd
from pathlib import Path

# ── 1. Cargar los tres archivos ──────────────────────────────────────

archivos = [
    "data/raw/catalogos_por_modelos.xlsx",
    "data/raw/catalogo_sudamericana.xlsx",
    "data/raw/catalogo_mundial.xlsx",
]

dfs = []
for archivo in archivos:
    df = pd.read_excel(archivo)
    dfs.append(df)
    print(f"Cargado: {Path(archivo).name} → {len(df)} filas")

# ── 2. Unir todo en un solo DataFrame ───────────────────────────────

catalogo = pd.concat(dfs, ignore_index=True)
catalogo.columns = catalogo.columns.str.strip()

print(f"\nTotal antes de limpiar: {len(catalogo)} filas")

# ── 3. Eliminar duplicados por Número de parte ───────────────────────

catalogo = catalogo.drop_duplicates(subset=["Número de parte"])
print(f"Total después de eliminar duplicados: {len(catalogo)} filas")

# ── 4. Limpiar valores nulos ─────────────────────────────────────────

catalogo["Descripción de la pieza "] = catalogo["Descripción de la pieza "].fillna("")
catalogo["Sistema principal"]         = catalogo["Sistema principal"].fillna("")
catalogo["Sub sistema de pertenencia"] = catalogo["Sub sistema de pertenencia"].fillna("")

# ── 5. Crear el texto que vamos a vectorizar ─────────────────────────

def crear_texto(fila):
    descripcion = str(fila["Descripción de la pieza "]).strip()
    sistema     = str(fila["Sistema principal"]).strip()
    subsistema  = str(fila["Sub sistema de pertenencia"]).strip()
    return f"{descripcion} - {sistema} - {subsistema}"

catalogo["texto_vectorizar"] = catalogo.apply(crear_texto, axis=1)

# ── 6. Ver los primeros resultados ───────────────────────────────────

print("\nEjemplos de texto a vectorizar:")
for texto in catalogo["texto_vectorizar"].head(5):
    print(f"  → {texto}")

    # ── 7. Vectorizar con HuggingFace ────────────────────────────────────

from sentence_transformers import SentenceTransformer
import chromadb

print("\nCargando modelo de embeddings...")
modelo = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

textos = catalogo["texto_vectorizar"].tolist()

print(f"Vectorizando {len(textos)} piezas...")
vectores = modelo.encode(textos, show_progress_bar=True)
print("Vectorización completa.")

# ── 8. Guardar en ChromaDB ───────────────────────────────────────────

print("\nGuardando en ChromaDB...")

cliente = chromadb.PersistentClient(path="data/chromadb")

# Eliminar colección si ya existe (para poder re-ejecutar limpio)
try:
    cliente.delete_collection("repuestos")
except:
    pass

coleccion = cliente.create_collection(
    name="repuestos",
    metadata={"hnsw:space": "cosine"}
)

coleccion.add(
    ids=catalogo["Número de parte"].astype(str).tolist(),
    embeddings=vectores.tolist(),
    documents=textos,
    metadatas=[
        {
            "numero_parte": str(row["Número de parte"]),
            "descripcion":  str(row["Descripción de la pieza"]),
            "sistema":      str(row["Sistema principal"]),
            "subsistema":   str(row["Sub sistema de pertenencia"]),
            "cantidad":     str(row["Cantidad necesaria"]),
        }
        for _, row in catalogo.iterrows()
    ]
)

print(f"✅ {coleccion.count()} piezas guardadas en ChromaDB.")
print("📁 Base de datos guardada en: data/chromadb/")