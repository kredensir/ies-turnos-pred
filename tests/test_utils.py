"""Pruebas básicas de carga, validación y limpieza."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from utils import validate_raw_data  # noqa: E402


def sample_dataframe() -> pd.DataFrame:
    """Devuelve un dataset mínimo con las columnas obligatorias."""
    return pd.DataFrame(
        {
            "PatientId": [1.0, 2.0, 3.0],
            "AppointmentID": [10, 11, 12],
            "Gender": ["F", "M", "F"],
            "ScheduledDay": [
                "2016-04-29T10:00:00Z",
                "2016-04-28T10:00:00Z",
                "2016-04-30T10:00:00Z",
            ],
            "AppointmentDay": [
                "2016-04-29T00:00:00Z",
                "2016-04-30T00:00:00Z",
                "2016-04-29T00:00:00Z",  # demora negativa
            ],
            "Age": [30, 40, -1],
            "Neighbourhood": ["A", "B", "A"],
            "Scholarship": [0, 1, 0],
            "Hipertension": [0, 0, 1],
            "Diabetes": [0, 1, 0],
            "Alcoholism": [0, 0, 0],
            "Handcap": [0, 1, 0],
            "SMS_received": [0, 1, 0],
            "No-show": ["No", "Yes", "No"],
        }
    )


class TestValidacion(unittest.TestCase):
    def test_reporte_basico(self) -> None:
        df = sample_dataframe()
        report = validate_raw_data(df, path=Path("test.csv"))

        self.assertEqual(report["shape"]["filas"], 3)
        self.assertEqual(report["shape"]["columnas"], 14)
        self.assertAlmostEqual(report["target"]["tasa_no_show"], 1 / 3, places=6)
        self.assertEqual(report["demora"]["negativas"], 1)
        self.assertEqual(report["edad"]["negativas"], 1)


class TestLimpieza(unittest.TestCase):
    def test_limpieza_corrige_casos_invalidos(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "limpieza",
            SRC / "02_limpieza.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        import logging

        logger = logging.getLogger("test_limpieza")
        logger.addHandler(logging.NullHandler())

        clean, log_entries = module.clean_data(sample_dataframe(), logger)

        self.assertEqual(clean["no_show"].tolist(), [0, 1, 0])
        self.assertGreaterEqual(clean["Age"].min(), 0)
        self.assertTrue(clean["Age"].between(0, 100).all())
        self.assertTrue((clean["waiting_days"] >= 0).all())
        self.assertEqual(clean["waiting_days_corrected"].sum(), 1)
        self.assertEqual(clean["age_imputed"].sum(), 1)
        self.assertIn("no_show", clean.columns)
        self.assertGreaterEqual(len(log_entries), 1)


if __name__ == "__main__":
    unittest.main()
