"""Prueba de integracion de la app Streamlit y del modelo final."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "app.py"
METADATA = ROOT / "models" / "model_metadata.json"


class TestAppStreamlit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not METADATA.exists():
            raise unittest.SkipTest(
                "No existe models/model_metadata.json; ejecuta src/05_modelado.py."
            )
        cls.metadata = json.loads(METADATA.read_text(encoding="utf-8"))

    def test_app_carga_y_predice(self) -> None:
        at = AppTest.from_file(str(APP)).run(timeout=180)
        self.assertEqual([str(e) for e in at.exception], [])

        at.button[0].click().run(timeout=180)
        self.assertEqual([str(e) for e in at.exception], [])

        etiquetas = [metrica.label for metrica in at.metric]
        self.assertIn("Probabilidad de no-show", etiquetas)
        self.assertIn("Umbral operativo", etiquetas)

        umbral = self.metadata["threshold"]
        self.assertGreaterEqual(umbral, 0.0)
        self.assertLessEqual(umbral, 1.0)


if __name__ == "__main__":
    unittest.main()
