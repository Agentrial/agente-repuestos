import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="angeldeveloper256/agente-repuestos-chromadb",
    repo_type="dataset",
    local_dir="data/chromadb",
    token=os.environ["HF_TOKEN"]
)
print("ChromaDB descargado correctamente.")
