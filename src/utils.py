"""
Utilidades compartidas del proyecto integrador.

Contiene rutas, configuración de logging, carga del dataset crudo,
validación inicial y persistencia de reportes.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Evita caracteres corruptos en consolas Windows cuando se imprimen acentos.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "KaggleV2-May-2016.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figuras"
LOGS_DIR = REPORTS_DIR / "logs"

# ---------------------------------------------------------------------------
# Constantes del dataset
# ---------------------------------------------------------------------------

TARGET_RAW = "No-show"
TARGET = "no_show"
TARGET_MAP = {"No": 0, "Yes": 1}

REQUIRED_COLUMNS = [
    "PatientId",
    "AppointmentID",
    "Gender",
    "ScheduledDay",
    "AppointmentDay",
    "Age",
    "Neighbourhood",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
    "No-show",
]

BINARY_COLUMNS = [
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "SMS_received",
]

DAY_NAMES_ES = {
    0: "Lunes",
    1: "Martes",
    2: "Miércoles",
    3: "Jueves",
    4: "Viernes",
    5: "Sábado",
    6: "Domingo",
}


class DataValidationError(Exception):
    """Error controlado cuando los datos no cumplen una regla de negocio."""


# ---------------------------------------------------------------------------
# Infraestructura
# ---------------------------------------------------------------------------

def ensure_directories() -> None:
    """Crea las carpetas de salida si no existen."""
    for path in (PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def configure_logging(name: str) -> logging.Logger:
    """Configura logging a consola y archivo, sin duplicar handlers."""
    ensure_directories()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(
        LOGS_DIR / f"{name}.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def _to_native(value: Any) -> Any:
    """Convierte tipos de pandas/numpy a tipos serializables en JSON."""
    if isinstance(value, dict):
        return {str(key): _to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def save_json(path: Path, data: Any) -> None:
    """Guarda un objeto en JSON UTF-8, creando la carpeta si hace falta."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_to_native(data), file, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Carga y validación
# ---------------------------------------------------------------------------

def load_raw_data(
    path: Path = RAW_PATH,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Carga el CSV crudo y valida que existan las columnas obligatorias."""
    logger = logger or logging.getLogger(__name__)

    try:
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo: {path}. "
                "Copiá KaggleV2-May-2016.csv en data/raw/."
            )

        df = pd.read_csv(path)

    except FileNotFoundError as exc:
        logger.error("Archivo no encontrado: %s", exc)
        raise
    except pd.errors.EmptyDataError as exc:
        logger.error("El archivo está vacío o no tiene columnas: %s", exc)
        raise
    except pd.errors.ParserError as exc:
        logger.error("El CSV no se pudo interpretar correctamente: %s", exc)
        raise
    except UnicodeDecodeError as exc:
        logger.error("Problema de codificación al leer el CSV: %s", exc)
        raise
    else:
        if df.empty:
            raise DataValidationError("El CSV se cargó, pero no contiene filas.")

        missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
        if missing:
            raise DataValidationError(
                "Faltan columnas obligatorias en el dataset: "
                + ", ".join(missing)
            )

        logger.info(
            "Datos cargados: %s filas x %s columnas desde %s",
            f"{len(df):,}",
            df.shape[1],
            path,
        )
        return df
    finally:
        logger.info("Fin de la etapa de carga.")


def validate_raw_data(
    df: pd.DataFrame,
    logger: logging.Logger | None = None,
    path: Path = RAW_PATH,
) -> dict[str, Any]:
    """Genera un diagnóstico de calidad del dataset crudo."""
    logger = logger or logging.getLogger(__name__)

    if df.empty:
        raise DataValidationError("No se puede validar un dataset vacío.")

    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise DataValidationError(
            "Faltan columnas obligatorias: " + ", ".join(missing)
        )

    duplicated_appointments = int(df["AppointmentID"].duplicated().sum())
    if duplicated_appointments:
        raise DataValidationError(
            f"Hay {duplicated_appointments} AppointmentID duplicados."
        )

    target_values = set(df[TARGET_RAW].dropna().astype(str).unique())
    invalid_target = target_values - set(TARGET_MAP)
    if invalid_target:
        raise DataValidationError(
            f"La columna {TARGET_RAW} contiene valores no esperados: "
            f"{sorted(invalid_target)}"
        )

    scheduled = pd.to_datetime(df["ScheduledDay"], errors="coerce")
    appointment = pd.to_datetime(df["AppointmentDay"], errors="coerce")
    invalid_scheduled = int(scheduled.isna().sum())
    invalid_appointment = int(appointment.isna().sum())

    waiting_days = (
        appointment.dt.normalize() - scheduled.dt.normalize()
    ).dt.days

    age = pd.to_numeric(df["Age"], errors="coerce")

    target_counts = (
        df[TARGET_RAW]
        .value_counts(dropna=False)
        .rename_axis("valor")
        .reset_index(name="cantidad")
    )
    target_dict = {
        str(row["valor"]): int(row["cantidad"])
        for _, row in target_counts.iterrows()
    }
    total = len(df)
    target_rate = (
        float(df[TARGET_RAW].eq("Yes").mean()) if total else float("nan")
    )

    report: dict[str, Any] = {
        "archivo": str(path),
        "shape": {"filas": int(df.shape[0]), "columnas": int(df.shape[1])},
        "columnas": df.columns.tolist(),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "memoria_mb": round(float(df.memory_usage(deep=True).sum() / 1024**2), 2),
        "nulos": {str(k): int(v) for k, v in df.isna().sum().items()},
        "duplicados_exactos": int(df.duplicated().sum()),
        "appointment_id_duplicados": duplicated_appointments,
        "target": {
            "conteo": target_dict,
            "tasa_no_show": round(target_rate, 6),
        },
        "edad": {
            "min": _to_native(age.min()),
            "max": _to_native(age.max()),
            "media": round(float(age.mean()), 3),
            "mediana": float(age.median()),
            "negativas": int((age < 0).sum()),
            "cero": int((age == 0).sum()),
            "mayores_100": int((age > 100).sum()),
        },
        "fechas": {
            "scheduled_min": _to_native(scheduled.min()),
            "scheduled_max": _to_native(scheduled.max()),
            "appointment_min": _to_native(appointment.min()),
            "appointment_max": _to_native(appointment.max()),
            "scheduled_invalidas": invalid_scheduled,
            "appointment_invalidas": invalid_appointment,
        },
        "demora": {
            "min": _to_native(waiting_days.min()),
            "max": _to_native(waiting_days.max()),
            "media": round(float(waiting_days.mean()), 3),
            "mediana": float(waiting_days.median()),
            "negativas": int((waiting_days < 0).sum()),
            "mismo_dia": int((waiting_days == 0).sum()),
        },
        "unicos": {
            "pacientes": int(df["PatientId"].nunique()),
            "turnos": int(df["AppointmentID"].nunique()),
            "barrios": int(df["Neighbourhood"].nunique()),
            "fechas_de_cita": int(appointment.dt.date.nunique()),
        },
        "genero": {
            str(k): int(v)
            for k, v in df["Gender"].value_counts(dropna=False).items()
        },
        "handcap": {
            str(k): int(v)
            for k, v in df["Handcap"].value_counts(dropna=False).sort_index().items()
        },
    }

    logger.info("=" * 68)
    logger.info("DIAGNÓSTICO INICIAL DEL DATASET")
    logger.info("=" * 68)
    logger.info(
        "Filas: %s | Columnas: %s | Memoria: %.2f MB",
        f"{report['shape']['filas']:,}",
        report["shape"]["columnas"],
        report["memoria_mb"],
    )
    logger.info(
        "Target: No=%s | Yes=%s | Tasa no-show=%.2f%%",
        f"{target_dict.get('No', 0):,}",
        f"{target_dict.get('Yes', 0):,}",
        target_rate * 100,
    )
    logger.info(
        "Nulos: %s | Duplicados exactos: %s | AppointmentID duplicados: %s",
        sum(report["nulos"].values()),
        report["duplicados_exactos"],
        duplicated_appointments,
    )
    logger.info(
        "Edad: min=%s | max=%s | negativas=%s | cero=%s | >100=%s",
        report["edad"]["min"],
        report["edad"]["max"],
        report["edad"]["negativas"],
        report["edad"]["cero"],
        report["edad"]["mayores_100"],
    )
    logger.info(
        "Demora: min=%s | max=%s | negativas=%s | mismo día=%s",
        report["demora"]["min"],
        report["demora"]["max"],
        report["demora"]["negativas"],
        report["demora"]["mismo_dia"],
    )
    logger.info(
        "Únicos: %s pacientes | %s barrios | %s fechas de cita",
        f"{report['unicos']['pacientes']:,}",
        report["unicos"]["barrios"],
        report["unicos"]["fechas_de_cita"],
    )
    logger.info("=" * 68)

    return report
