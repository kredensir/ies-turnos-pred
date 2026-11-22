#!/usr/bin/env python
"""
Etapa 5: ingeniería de características y partición de datos.

Estrategia principal:
    - Separar test (15 %) y validation (15 %) con GroupShuffleSplit por PatientId.
    - Evitar que un mismo paciente aparezca en más de un conjunto.
    - Guardar train/validation/test y la especificación de features.

Uso:
    python src/03_features.py

Entradas:
    data/processed/turnos_clean.parquet (o .csv)

Salidas:
    data/processed/train.csv / validation.csv / test.csv
    data/processed/train.parquet / validation.parquet / test.parquet
    reports/split_summary.json
    reports/feature_list.json
    reports/logs/03_features.log
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from utils import (
    PROCESSED_DIR,
    REPORTS_DIR,
    TARGET,
    DataValidationError,
    configure_logging,
    save_json,
)


ID_COLUMNS = ["PatientId", "AppointmentID"]
TARGET_COLUMN = TARGET

NUMERIC_FEATURES = [
    "Age",
    "waiting_days",
    "scheduled_hour",
    "scheduled_weekday",
    "appointment_weekday",
    "appointment_week",
    "appointment_month",
    "is_weekend",
    "chronic_conditions",
    "has_chronic_condition",
    "handicap_any",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
    "is_same_day",
    "age_imputed",
    "age_capped",
    "waiting_days_corrected",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Neighbourhood",
    "age_group",
    "waiting_days_bucket",
]

DROP_COLUMNS = [
    "PatientId",
    "AppointmentID",
    "ScheduledDay",
    "AppointmentDay",
    "scheduled_date",
    "appointment_date",
    "appointment_weekday_name",
    TARGET_COLUMN,
]


def load_clean_data(logger) -> pd.DataFrame:
    """Carga el dataset limpio desde Parquet o, si no existe, desde CSV."""
    parquet_path = PROCESSED_DIR / "turnos_clean.parquet"
    csv_path = PROCESSED_DIR / "turnos_clean.csv"

    try:
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            source = parquet_path
        elif csv_path.exists():
            df = pd.read_csv(
                csv_path,
                parse_dates=["ScheduledDay", "AppointmentDay"],
            )
            source = csv_path
        else:
            raise FileNotFoundError(
                "No existe turnos_clean.parquet ni turnos_clean.csv. "
                "Ejecutá primero src/02_limpieza.py."
            )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        raise
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        logger.error("No se pudo leer el dataset limpio: %s", exc)
        raise
    else:
        required = set(
            ID_COLUMNS + NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
        )
        missing = sorted(required - set(df.columns))
        if missing:
            raise DataValidationError(
                "Faltan columnas en el dataset limpio: " + ", ".join(missing)
            )
        logger.info("Dataset limpio cargado desde %s: %s filas", source, f"{len(df):,}")
        return df
    finally:
        logger.info("Fin de la carga del dataset limpio.")


def split_by_patient(
    df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide 70/15/15 aproximadamente, sin repetir pacientes entre splits."""
    if df["PatientId"].isna().any():
        raise DataValidationError("PatientId contiene nulos; no se puede agrupar.")

    test_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=random_state,
    )
    train_val_idx, test_idx = next(
        test_splitter.split(df, groups=df["PatientId"])
    )

    train_val = df.iloc[train_val_idx].copy()
    test = df.iloc[test_idx].copy()

    val_fraction = 0.15 / 0.85
    val_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_fraction,
        random_state=random_state,
    )
    train_idx, val_idx = next(
        val_splitter.split(train_val, groups=train_val["PatientId"])
    )

    train = train_val.iloc[train_idx].copy()
    validation = train_val.iloc[val_idx].copy()
    return train, validation, test


def validate_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    original: pd.DataFrame,
) -> dict:
    """Verifica que no haya fuga de pacientes y devuelve un resumen."""
    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    total_rows = sum(len(part) for part in splits.values())
    if total_rows != len(original):
        raise DataValidationError(
            "La suma de filas de los splits no coincide con el dataset original."
        )

    train_patients = set(train["PatientId"])
    val_patients = set(validation["PatientId"])
    test_patients = set(test["PatientId"])

    if train_patients & val_patients:
        raise DataValidationError("Hay pacientes compartidos entre train y validation.")
    if train_patients & test_patients:
        raise DataValidationError("Hay pacientes compartidos entre train y test.")
    if val_patients & test_patients:
        raise DataValidationError("Hay pacientes compartidos entre validation y test.")

    all_patients = set(original["PatientId"])
    if (train_patients | val_patients | test_patients) != all_patients:
        raise DataValidationError("La unión de pacientes no cubre el dataset original.")

    summary = {
        "total_filas": int(len(original)),
        "total_pacientes": int(original["PatientId"].nunique()),
        "splits": {},
        "sin_fuga_de_pacientes": True,
    }

    for name, part in splits.items():
        summary["splits"][name] = {
            "filas": int(len(part)),
            "porcentaje_filas": round(len(part) / len(original) * 100, 2),
            "pacientes": int(part["PatientId"].nunique()),
            "tasa_no_show": round(float(part[TARGET_COLUMN].mean()), 6),
            "fecha_min": str(part["AppointmentDay"].min()),
            "fecha_max": str(part["AppointmentDay"].max()),
        }

    return summary


def main() -> int:
    logger = configure_logging("03_features")
    logger.info("=" * 68)
    logger.info("ETAPA 5: FEATURES Y PARTICIÓN DE DATOS")
    logger.info("=" * 68)

    try:
        df = load_clean_data(logger=logger)
        train, validation, test = split_by_patient(df, random_state=42)
        summary = validate_split(train, validation, test, df)

        for name, part in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        ):
            csv_path = PROCESSED_DIR / f"{name}.csv"
            part.to_csv(csv_path, index=False, encoding="utf-8-sig")
            logger.info("Guardado %s: %s filas -> %s", name, f"{len(part):,}", csv_path)

            try:
                part.to_parquet(
                    PROCESSED_DIR / f"{name}.parquet",
                    index=False,
                )
            except (ImportError, ModuleNotFoundError, ValueError) as exc:
                logger.warning("No se pudo guardar %s.parquet: %s", name, exc)

        feature_spec = {
            "target": TARGET_COLUMN,
            "id_columns": ID_COLUMNS,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "drop_columns": DROP_COLUMNS,
            "nota": (
                "Las columnas de fecha y los identificadores se conservan en los "
                "archivos para trazabilidad y EDA, pero no se usan como features. "
                "PatientId se usa solo para la partición por paciente."
            ),
        }

        save_json(REPORTS_DIR / "split_summary.json", summary)
        save_json(REPORTS_DIR / "feature_list.json", feature_spec)

        logger.info("Resumen de la partición:")
        for name, info in summary["splits"].items():
            logger.info(
                "  %-10s | filas=%s (%.2f%%) | pacientes=%s | no-show=%.2f%%",
                name,
                f"{info['filas']:,}",
                info["porcentaje_filas"],
                f"{info['pacientes']:,}",
                info["tasa_no_show"] * 100,
            )
        logger.info("Sin fuga de pacientes entre splits: OK")
        logger.info("La etapa 5 finalizó correctamente.")
        return 0

    except (
        FileNotFoundError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        DataValidationError,
    ) as exc:
        logger.error("No se pudo completar la partición: %s", exc)
        return 1

    finally:
        logger.info("Fin del script 03_features.py")


if __name__ == "__main__":
    raise SystemExit(main())
