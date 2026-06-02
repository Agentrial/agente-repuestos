# cotizador-mlops

Sistema de cotización inteligente basado en un grafo de conocimiento (Knowledge Graph) que conecta servicios técnicos, síntomas, piezas y precios históricos. El pipeline de MLOps permite ingestar datos brutos, enriquecer precios y construir el grafo para alimentar modelos de predicción y cotización automática.

## Arquitectura

```
cotizador-mlops/
├── data/
│   ├── raw/               # CSVs / JSONs originales sin transformar
│   ├── processed/         # Datos limpios listos para el grafo
│   └── enriched/          # Datos con precios enriquecidos (APIs externas, inflación, etc.)
├── scripts/
│   ├── ingest.py          # Ingesta y validación de datos brutos
│   └── enrich_prices.py   # Enriquecimiento de precios con fuentes externas
├── src/
│   └── knowledge_graph/
│       └── graph_builder.py  # Construcción del grafo: nodos y aristas
├── notebooks/             # Exploración y prototipado
├── tests/                 # Tests unitarios y de integración
├── .env.example
├── requirements.txt
└── README.md
```

## Grafo de conocimiento

El grafo tiene tres tipos de nodos y sus aristas:

| Nodo       | Descripción                                      |
|------------|--------------------------------------------------|
| `Servicio` | Servicio técnico ofrecido (e.g. cambio de aceite)|
| `Síntoma`  | Síntoma reportado por el cliente                 |
| `Pieza`    | Pieza física involucrada en el servicio          |

| Arista               | Origen     | Destino    | Descripción                              |
|----------------------|------------|------------|------------------------------------------|
| `REQUIERE_PIEZA`     | Servicio   | Pieza      | El servicio necesita esta pieza          |
| `INDICA_SERVICIO`    | Síntoma    | Servicio   | El síntoma sugiere este servicio         |
| `ASOCIADO_A`         | Síntoma    | Pieza      | El síntoma está relacionado con la pieza |

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/Agentrial/cotizador-mlops.git
cd cotizador-mlops

# Crear y activar entorno virtual (Python 3.14, WSL2 Ubuntu)
python3.14 -m venv ~/proyectos/cotizador-mlops
source ~/proyectos/cotizador-mlops/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales
```

## Pipeline de uso

### 1. Ingestar datos brutos

```bash
python scripts/ingest.py \
  --input data/raw/servicios.csv \
  --output data/processed/servicios_clean.json
```

### 2. Enriquecer precios

```bash
python scripts/enrich_prices.py \
  --input data/processed/servicios_clean.json \
  --output data/enriched/servicios_enriched.json
```

### 3. Construir el grafo

```python
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder

builder = KnowledgeGraphBuilder()
builder.load_from_file("data/enriched/servicios_enriched.json")
builder.build()
builder.export("data/graph/cotizador.graphml")

# Consultar el grafo
servicios = builder.get_servicios_for_sintoma("ruido al frenar")
print(servicios)
```

## Variables de entorno

Ver `.env.example` para la lista completa. Las principales:

| Variable              | Descripción                                    |
|-----------------------|------------------------------------------------|
| `NEO4J_URI`           | URI de conexión a Neo4j (opcional)             |
| `NEO4J_USER`          | Usuario Neo4j                                  |
| `NEO4J_PASSWORD`      | Contraseña Neo4j                               |
| `PRICE_API_KEY`       | API key para enriquecimiento de precios        |
| `LOG_LEVEL`           | Nivel de logging (`INFO`, `DEBUG`, `WARNING`)  |

## Tests

```bash
pytest tests/ -v
```

## Contribuir

1. Crear rama: `git checkout -b feat/nombre-feature`
2. Commitear cambios con mensajes descriptivos
3. Abrir Pull Request hacia `main`

## Licencia

MIT