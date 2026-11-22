#!/usr/bin/env python
"""
Etapa 3: limpieza y preparación del dataset.

Uso:
    python src/02_limpieza.py

Entrada:
    data/raw/KaggleV2-May-2016.csv

Salidas:
    data/processed/turnos_clean.csv
    data/processed/turnos_clean.parquet
    reports/limpieza_log.csv
    reports/resumen_limpieza.json
    reports/logs/02_limpieza.log
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from utils import (
    BINARY_COLUMNS,
    DAY_NAMES_ES,
    PROCESSED_DIR,
    REPORTS_DIR,
    TARGET,
    TARGET_MAP,
    TARGET_RAW,
    DataValidationError,
    configure_logging,
    load_raw_data,
    save_json,
)


def _add_log(entries: list[dict], rule: str, affected: int, action: str) -> None:
    entries.append(
        {
            "regla": rule,
            "filas_afectadas": int(affected),
            "accion": action,
        }
    )


def clean_data(df: pd.DataFrame, logger) -> tuple[pd.DataFrame, list[dict]]:
    """Aplica las reglas de limpieza y devuelve el dataset y el log."""
    log_entries: list[dict] = []
    rows_before = len(df)
    cols_before = df.shape[1]

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    # ------------------------------------------------------------------
    # 1. Duplicados exactos y por identificador de turno
    # ------------------------------------------------------------------
    exact_duplicates = int(df.duplicated().sum())
    if exact_duplicates:
        df = df.drop_duplicates(keep="first").copy()
    _add_log(
        log_entries,
        "Duplicados exactos",
        exact_duplicates,
        "Se conserva la primera aparición." if exact_duplicates else "Sin cambios.",
    )

    appointment_duplicates = int(df["AppointmentID"].duplicated().sum())
    if appointment_duplicates:
        df = df.drop_duplicates(subset=["AppointmentID"], keep="first").copy()
    _add_log(
        log_entries,
        "AppointmentID duplicados",
        appointment_duplicates,
        "Se conserva un registro por turno." if appointment_duplicates else "Sin cambios.",
    )

    if df["AppointmentID"].isna().any():
        raise DataValidationError("Hay AppointmentID nulos; no se puede continuar.")

    if df["PatientId"].isna().any():
        raise DataValidationError("Hay PatientId nulos; no se puede continuar.")
    df["PatientId"] = df["PatientId"].astype("int64")
    _add_log(
        log_entries,
        "PatientId convertido a entero",
        len(df),
        "Se conserva para la partición por paciente; no se usará como feature.",
    )

    # ------------------------------------------------------------------
    # 2. Normalización de texto
    # ------------------------------------------------------------------
    gender_before = df["Gender"].astype("string").str.strip().str.upper()
    gender_invalid = int((~gender_before.isin(["F", "M"])).sum())
    df["Gender"] = gender_before.fillna("Desconocido")
    _add_log(
        log_entries,
        "Gender normalizado",
        gender_invalid,
        "Se estandariza a F/M; valores inválidos pasan a 'Desconocido'.",
    )

    neighbourhood_before = df["Neighbourhood"].astype("string")
    neighbourhood_clean = (
        neighbourhood_before.str.replace(r"\s+", " ", regex=True).str.strip()
    )
    neighbourhood_changed = int(
        (neighbourhood_before.fillna("") != neighbourhood_clean.fillna("")).sum()
    )
    neighbourhood_missing = int(neighbourhood_clean.isna().sum())
    df["Neighbourhood"] = neighbourhood_clean.fillna("Desconocido")
    _add_log(
        log_entries,
        "Neighbourhood normalizado",
        neighbourhood_changed + neighbourhood_missing,
        "Se eliminan espacios extra; nulos pasan a 'Desconocido'.",
    )

    # ------------------------------------------------------------------
    # 3. Variables binarias y ordinales
    # ------------------------------------------------------------------
    for column in BINARY_COLUMNS:
        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid = numeric.isna() | (~numeric.isin([0, 1]))
        invalid_count = int(invalid.sum())
        if invalid_count:
            mode = numeric.mode(dropna=True)
            fill_value = int(mode.iloc[0]) if not mode.empty else 0
            numeric = numeric.mask(invalid, fill_value)
            _add_log(
                log_entries,
                f"{column} inválido",
                invalid_count,
                f"Se imputa con la moda ({fill_value}).",
            )
        df[column] = numeric.astype(int)

    handcap_numeric = pd.to_numeric(df["Handcap"], errors="coerce")
    handcap_invalid = handcap_numeric.isna() | (~handcap_numeric.isin([0, 1, 2, 3, 4]))
    handcap_invalid_count = int(handcap_invalid.sum())
    if handcap_invalid_count:
        mode = handcap_numeric.mode(dropna=True)
        fill_value = int(mode.iloc[0]) if not mode.empty else 0
        handcap_numeric = handcap_numeric.mask(handcap_invalid, fill_value)
        _add_log(
            log_entries,
            "Handcap inválido",
            handcap_invalid_count,
            f"Se imputa con la moda ({fill_value}) y se limita a 0–4.",
        )
    df["Handcap"] = handcap_numeric.clip(0, 4).astype(int)

    # ------------------------------------------------------------------
    # 4. Edad
    # ------------------------------------------------------------------
    age_numeric = pd.to_numeric(df["Age"], errors="coerce")
    age_invalid = age_numeric.isna() | (age_numeric < 0)
    age_invalid_count = int(age_invalid.sum())
    median_age = float(age_numeric[~age_invalid].median())
    if pd.isna(median_age):
        raise DataValidationError("No se pudo calcular la mediana de Age.")
    df["age_imputed"] = age_invalid.astype(int)
    df["Age"] = age_numeric.mask(age_invalid, median_age).round().astype(int)
    _add_log(
        log_entries,
        "Age negativa o nula",
        age_invalid_count,
        f"Se imputa con la mediana ({median_age:.0f}) y se marca age_imputed.",
    )

    age_capped = df["Age"] > 100
    age_capped_count = int(age_capped.sum())
    df["age_capped"] = age_capped.astype(int)
    df["Age"] = df["Age"].clip(upper=100)
    _add_log(
        log_entries,
        "Age mayor a 100",
        age_capped_count,
        "Se limita a 100 y se marca age_capped.",
    )

    # ------------------------------------------------------------------
    # 5. Fechas y demora
    # ------------------------------------------------------------------
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"], errors="raise")
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], errors="raise")

    df["scheduled_date"] = df["ScheduledDay"].dt.date
    df["appointment_date"] = df["AppointmentDay"].dt.date

    df["waiting_days"] = (
        df["AppointmentDay"].dt.normalize() - df["ScheduledDay"].dt.normalize()
    ).dt.days.astype(int)

    negative_waiting = df["waiting_days"] < 0
    negative_waiting_count = int(negative_waiting.sum())
    df["waiting_days_corrected"] = negative_waiting.astype(int)
    df.loc[negative_waiting, "waiting_days"] = 0
    _add_log(
        log_entries,
        "Demora negativa",
        negative_waiting_count,
        "Se corrige a 0 y se marca waiting_days_corrected.",
    )

    # ------------------------------------------------------------------
    # 6. Variable objetivo
    # ------------------------------------------------------------------
    target_raw = df[TARGET_RAW].astype("string").str.strip()
    df[TARGET] = target_raw.map(TARGET_MAP)
    if df[TARGET].isna().any():
        invalid = sorted(target_raw[df[TARGET].isna()].dropna().unique().tolist())
        raise DataValidationError(
            f"Valores inesperados en {TARGET_RAW}: {invalid}"
        )
    df[TARGET] = df[TARGET].astype(int)
    df = df.drop(columns=[TARGET_RAW])
    _add_log(
        log_entries,
        "No-show mapeado",
        rows_before,
        "Yes=1, No=0; la columna original se reemplaza por no_show.",
    )

    # ------------------------------------------------------------------
    # 7. Variables derivadas para EDA y modelado
    # ------------------------------------------------------------------
    df["waiting_days_bucket"] = pd.cut(
        df["waiting_days"],
        bins=[-0.1, 0, 7, 14, 30, 10_000],
        labels=["0", "1-7", "8-14", "15-30", ">30"],
        include_lowest=True,
    )
    df["age_group"] = pd.cut(
        df["Age"],
        bins=[-0.1, 12, 18, 30, 45, 60, 200],
        labels=["0-12", "13-18", "19-30", "31-45", "46-60", "61+"],
        include_lowest=True,
    )
    df["is_same_day"] = (df["waiting_days"] == 0).astype(int)
    df["appointment_weekday"] = df["AppointmentDay"].dt.dayofweek.astype(int)
    df["appointment_weekday_name"] = df["appointment_weekday"].map(DAY_NAMES_ES)
    df["appointment_week"] = (
        df["AppointmentDay"].dt.isocalendar().week.astype(int)
    )
    df["appointment_month"] = df["AppointmentDay"].dt.month.astype(int)
    df["is_weekend"] = (df["appointment_weekday"] >= 5).astype(int)
    df["scheduled_weekday"] = df["ScheduledDay"].dt.dayofweek.astype(int)
    df["scheduled_hour"] = df["ScheduledDay"].dt.hour.astype(int)

    df["chronic_conditions"] = (
        df["Hipertension"] + df["Diabetes"] + df["Alcoholism"]
    ).astype(int)
    df["has_chronic_condition"] = (df["chronic_conditions"] > 0).astype(int)
    df["handicap_any"] = (df["Handcap"] > 0).astype(int)

    _add_log(
        log_entries,
        "Variables derivadas",
        10,
        (
            "waiting_days_bucket, age_group, is_same_day, appointment_weekday, "
            "appointment_week, appointment_month, is_weekend, scheduled_weekday, "
            "scheduled_hour, chronic_conditions, has_chronic_condition, handicap_any."
        ),
    )

    # ------------------------------------------------------------------
    # 8. Orden final de columnas
    # ------------------------------------------------------------------
    ordered_columns = [
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
        "scheduled_date",
        "appointment_date",
        "waiting_days",
        "waiting_days_bucket",
        "waiting_days_corrected",
        "is_same_day",
        "appointment_weekday",
        "appointment_weekday_name",
        "appointment_week",
        "appointment_month",
        "is_weekend",
        "scheduled_weekday",
        "scheduled_hour",
        "age_group",
        "age_imputed",
        "age_capped",
        "chronic_conditions",
        "has_chronic_condition",
        "handicap_any",
        "no_show",
    ]
    missing_columns = [col for col in ordered_columns if col not in df.columns]
    if missing_columns:
        raise DataValidationError(
            "Faltan columnas al finalizar la limpieza: "
            + ", ".join(missing_columns)
        )
    df = df[ordered_columns]

    # ------------------------------------------------------------------
    # 9. Validaciones posteriores
    # ------------------------------------------------------------------
    if df.empty:
        raise DataValidationError("El dataset limpio quedó vacío.")
    if df[TARGET].isna().any():
        raise DataValidationError("Hay valores nulos en no_show.")
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise DataValidationError("no_show debe contener solo 0 y 1.")
    if not df["Age"].between(0, 100).all():
        raise DataValidationError("Age quedó fuera del rango 0–100.")
    if (df["waiting_days"] < 0).any():
        raise DataValidationError("Quedaron demoras negativas.")
    if df["AppointmentID"].duplicated().any():
        raise DataValidationError("Quedaron AppointmentID duplicados.")
    if df["PatientId"].isna().any():
        raise DataValidationError("Hay PatientId nulos.")
    if df["Gender"].isna().any() or df["Neighbourhood"].isna().any():
        raise DataValidationError("Hay nulos en Gender o Neighbourhood.")

    logger.info(
        "Limpieza aplicada: %s filas -> %s filas | %s columnas -> %s columnas",
        f"{rows_before:,}",
        f"{len(df):,}",
        cols_before,
        df.shape[1],
    )
    logger.info(
        "Target final: No=%s | Yes=%s | Tasa no-show=%.2f%%",
        f"{int((df[TARGET] == 0).sum()):,}",
        f"{int((df[TARGET] == 1).sum()):,}",
        float(df[TARGET].mean() * 100),
    )

    return df, log_entries


def main() -> int:
    logger = configure_logging("02_limpieza")
    logger.info("=" * 68)
    logger.info("ETAPA 3: LIMPIEZA Y PREPARACIÓN DEL DATASET")
    logger.info("=" * 68)

    try:
        raw = load_raw_data(logger=logger)
        clean, log_entries = clean_data(raw, logger=logger)

        csv_path = PROCESSED_DIR / "turnos_clean.csv"
        parquet_path = PROCESSED_DIR / "turnos_clean.parquet"
        clean.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info("Dataset limpio guardado en: %s", csv_path)

        try:
            clean.to_parquet(parquet_path, index=False)
            logger.info("Versión Parquet guardada en: %s", parquet_path)
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            logger.warning(
                "No se pudo guardar Parquet (%s). Se conserva el CSV.",
                exc,
            )

        log_df = pd.DataFrame(log_entries)
        log_path = REPORTS_DIR / "limpieza_log.csv"
        log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
        logger.info("Log de limpieza guardado en: %s", log_path)

        target_counts = {
            str(key): int(value)
            for key, value in clean[TARGET].value_counts().sort_index().items()
        }
        summary = {
            "filas_antes": int(len(raw)),
            "filas_despues": int(len(clean)),
            "filas_eliminadas": int(len(raw) - len(clean)),
            "columnas_antes": int(raw.shape[1]),
            "columnas_despues": int(clean.shape[1]),
            "target": {
                "conteo": target_counts,
                "tasa_no_show": round(float(clean[TARGET].mean()), 6),
            },
            "nulos_totales": int(clean.isna().sum().sum()),
            "edad": {
                "min": int(clean["Age"].min()),
                "max": int(clean["Age"].max()),
                "media": round(float(clean["Age"].mean()), 3),
            },
            "demora": {
                "min": int(clean["waiting_days"].min()),
                "max": int(clean["waiting_days"].max()),
                "media": round(float(clean["waiting_days"].mean()), 3),
            },
            "unicos": {
                "pacientes": int(clean["PatientId"].nunique()),
                "barrios": int(clean["Neighbourhood"].nunique()),
            },
            "archivos": {
                "csv": str(csv_path),
                "parquet": str(parquet_path),
                "log": str(log_path),
            },
        }
        summary_path = REPORTS_DIR / "resumen_limpieza.json"
        save_json(summary_path, summary)
        logger.info("Resumen guardado en: %s", summary_path)
        logger.info("La etapa 3 finalizó correctamente.")

    except (
        FileNotFoundError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        DataValidationError,
    ) as exc:
        logger.error("No se pudo completar la limpieza: %s", exc)
        return 1
    else:
        return 0
    finally:
        logger.info("Fin del script 02_limpieza.py")


if __name__ == "__main__":
    raise SystemExit(main())
