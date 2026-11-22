#!/usr/bin/env python
"""
Etapa 2: carga y validación del dataset crudo.

Uso:
    python src/01_carga_validacion.py

Salidas:
    reports/validacion_inicial.json
    reports/logs/01_carga_validacion.log
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from utils import (
    REPORTS_DIR,
    DataValidationError,
    configure_logging,
    load_raw_data,
    save_json,
    validate_raw_data,
)


def main() -> int:
    logger = configure_logging("01_carga_validacion")
    logger.info("=" * 68)
    logger.info("ETAPA 2: CARGA Y VALIDACIÓN DEL DATASET")
    logger.info("=" * 68)

    try:
        df = load_raw_data(logger=logger)
        report = validate_raw_data(df, logger=logger)

    except (
        FileNotFoundError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        DataValidationError,
    ) as exc:
        logger.error("No se pudo completar la carga/validación: %s", exc)
        return 1

    else:
        output = REPORTS_DIR / "validacion_inicial.json"
        save_json(output, report)
        logger.info("Reporte de validación guardado en: %s", output)
        logger.info("La etapa 2 finalizó correctamente.")
        return 0

    finally:
        logger.info("Fin del script 01_carga_validacion.py")


if __name__ == "__main__":
    raise SystemExit(main())
