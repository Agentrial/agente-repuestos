"""
scripts/ingest.py
-----------------
Ingesta y validación de datos brutos hacia el directorio processed/.

Soporta CSV y JSON. Valida esquema mínimo, normaliza campos y exporta
registros limpios como JSON estructurado para las etapas siguientes.

Uso:
    python scripts/ingest.py --input data/raw/servicios.csv --output data/processed/servicios_clean.json
    python scripts/ingest.py --input data/raw/ --output data/processed/  # procesa carpeta completa
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ingest")

# ---------------------------------------------------------------------------
# Esquema mínimo esperado en los datos brutos
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {"servicio", "sintoma", "pieza", "precio"}

FIELD_ALIASES: dict[str, list[str]] = {
    "servicio": ["servicio", "service", "nombre_servicio", "service_name"],
    "sintoma":  ["sintoma", "síntoma", "symptom", "problema", "issue"],
    "pieza":    ["pieza", "parte", "part", "component", "componente"],
    "precio":   ["precio", "price", "costo", "cost", "monto"],
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def normalize_header(raw: str) -> str:
    """Convierte cabecera cruda al nombre canónico si existe alias."""
    key = raw.strip().lower().replace(" ", "_")
    for canonical, aliases in FIELD_ALIASES.items():
        if key in aliases:
            return canonical
    return key


def validate_record(record: dict[str, Any], index: int) -> list[str]:
    """Devuelve lista de errores de validación para un registro."""
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        value = record.get(field)
        if value is None or str(value).strip() == "":
            errors.append(f"Campo '{field}' vacío o ausente en registro #{index}")
    try:
        price = float(str(record.get("precio", "")).replace(",", "."))
        if price < 0:
            errors.append(f"Precio negativo ({price}) en registro #{index}")
    except ValueError:
        errors.append(f"Precio no numérico '{record.get('precio')}' en registro #{index}")
    return errors


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normaliza tipos y limpia espacios en un registro válido."""
    return {
        "servicio": str(record["servicio"]).strip().title(),
        "sintoma":  str(record["sintoma"]).strip().lower(),
        "pieza":    str(record["pieza"]).strip().title(),
        "precio":   float(str(record["precio"]).replace(",", ".")),
        # Campos opcionales
        "descripcion":  str(record.get("descripcion", "")).strip() or None,
        "categoria":    str(record.get("categoria", "")).strip() or None,
        "tiempo_horas": _parse_float(record.get("tiempo_horas")),
    }


def _parse_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Lectores
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict[str, Any]]:
    """Lee un CSV y normaliza cabeceras."""
    records: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV sin cabeceras: {path}")
        normalized_fields = [normalize_header(h) for h in reader.fieldnames]
        for row in reader:
            normalized = {
                normalize_header(k): v for k, v in row.items()
            }
            records.append(normalized)
    log.info("CSV leído: %d registros desde %s", len(records), path)
    return records


def read_json(path: Path) -> list[dict[str, Any]]:
    """Lee un JSON (lista de objetos o dict con clave 'data'/'records')."""
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict):
        for key in ("data", "records", "items", "servicios"):
            if key in raw and isinstance(raw[key], list):
                records = raw[key]
                break
        else:
            raise ValueError(f"JSON no contiene lista reconocible en {path}")
    else:
        raise ValueError(f"Formato JSON no soportado en {path}")

    # Normalizar claves
    normalized = [{normalize_header(k): v for k, v in r.items()} for r in records]
    log.info("JSON leído: %d registros desde %s", len(normalized), path)
    return normalized


def read_file(path: Path) -> list[dict[str, Any]]:
    """Despacha al lector correcto según extensión."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv(path)
    elif suffix == ".json":
        return read_json(path)
    else:
        raise ValueError(f"Formato no soportado: {suffix} ({path})")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def ingest(input_path: Path, output_path: Path, strict: bool = False) -> int:
    """
    Ejecuta la ingesta completa.

    Returns:
        Número de registros exportados con éxito.
    """
    # Recopilar archivos a procesar
    if input_path.is_dir():
        files = list(input_path.glob("*.csv")) + list(input_path.glob("*.json"))
        if not files:
            log.warning("No se encontraron archivos en %s", input_path)
            return 0
    else:
        files = [input_path]

    all_records: list[dict[str, Any]] = []
    total_errors = 0

    for file in files:
        log.info("Procesando: %s", file)
        try:
            raw_records = read_file(file)
        except Exception as exc:
            log.error("Error leyendo %s: %s", file, exc)
            if strict:
                raise
            continue

        for i, record in enumerate(raw_records, start=1):
            errors = validate_record(record, i)
            if errors:
                for err in errors:
                    log.warning("%s", err)
                total_errors += len(errors)
                if strict:
                    log.error("Modo strict: abortando por errores de validación.")
                    sys.exit(1)
                continue
            all_records.append(clean_record(record))

    log.info(
        "Ingesta completa: %d registros válidos, %d errores ignorados.",
        len(all_records), total_errors,
    )

    # Exportar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    log.info("Exportado: %s", output_path)
    return len(all_records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingesta y validación de datos brutos para cotizador-mlops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Archivo CSV/JSON de entrada o directorio con múltiples archivos.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Ruta del JSON de salida (o directorio si la entrada es un directorio).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abortar al primer error de validación en vez de omitir el registro.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_path: Path = args.input
    output_path: Path = args.output

    if not input_path.exists():
        log.error("La ruta de entrada no existe: %s", input_path)
        sys.exit(1)

    # Si la entrada es un directorio, la salida también debe serlo (o crearse)
    if input_path.is_dir() and not output_path.suffix:
        output_path = output_path / "ingested.json"

    count = ingest(input_path, output_path, strict=args.strict)
    if count == 0:
        log.warning("No se exportaron registros.")
        sys.exit(2)


if __name__ == "__main__":
    main()