"""
src/llm/gemini_client.py
------------------------
Cliente para Gemini. Lee configuración desde config/prompts.yaml.
El código no cambia cuando cambia el prompt o el modelo.
"""

import os
import yaml
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ── Cargar configuración desde YAML ─────────────────────────────────

def _cargar_config() -> dict:
    ruta = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"
    with ruta.open(encoding="utf-8") as f:
        return yaml.safe_load(f)

CONFIG = _cargar_config()

# ── Construir system prompt desde config ────────────────────────────

def _construir_prompt(cfg: dict) -> str:
    sp = cfg["system_prompt"]
    restricciones = "\n".join(f"- {r}" for r in sp["restricciones"])
    return f"""
ROL:
{sp["rol"]}

IDIOMA: {sp["idioma"]}
TONO: {sp["tono"]}

FORMATO DE RESPUESTA:
{sp["formato_respuesta"]}

RESTRICCIONES:
{restricciones}
""".strip()

SYSTEM_PROMPT = _construir_prompt(CONFIG)

# ── Cliente Gemini ───────────────────────────────────────────────────

cliente = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── Función principal ────────────────────────────────────────────────

def consultar(pregunta: str, repuestos: list[dict]) -> tuple:
    contexto = "\n".join([
        f"- {r['descripcion']} #{r['numero_parte']}"
        for r in repuestos
    ])

    prompt = f"""{SYSTEM_PROMPT}

CONSULTA DEL TÉCNICO:
{pregunta}

REPUESTOS DEL CATÁLOGO:
{contexto}"""

    model_cfg = CONFIG["model"]

    respuesta = cliente.models.generate_content(
        model=model_cfg["nombre"],
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=model_cfg["temperature"],
            max_output_tokens=model_cfg["max_output_tokens"],
            thinking_config=types.ThinkingConfig(
                thinking_budget=0
            )
        )
    )

    finish = respuesta.candidates[0].finish_reason
    uso    = respuesta.usage_metadata

    return respuesta.text, uso, finish