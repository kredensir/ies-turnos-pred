#!/usr/bin/env python
"""
Etapa 6: modelado, evaluacion y seleccion de umbral.

Objetivo:
    - Entrenar un baseline y tres modelos supervisados para predecir no-show.
    - Medir tiempo de ajuste, tiempo de inferencia y memoria RAM.
    - Evaluar con metricas adecuadas para clases desbalanceadas.
    - Elegir un umbral operativo con una regla de costo explicita.
    - Guardar el modelo final y los metadatos que usara la app Streamlit.

Modelos:
    - DummyClassifier (baseline sin aprendizaje).
    - LogisticRegression (modelo lineal interpretable).
    - RandomForestClassifier (ensemble de arboles).
    - HistGradientBoostingClassifier (boosting eficiente).

Uso:
    python src/05_modelado.py

Entradas:
    data/processed/train.parquet
    data/processed/validation.parquet
    data/processed/test.parquet
    reports/feature_list.json

Salidas:
    models/*.joblib
    models/final_model.joblib
    models/model_metadata.json
    reports/model_comparison.csv
    reports/model_metrics.json
    reports/threshold_analysis.csv
    reports/threshold_grid.csv
    reports/model_card.md
    reports/figuras/11_model_comparison.png
    reports/figuras/12_roc_curves.png
    reports/figuras/13_pr_curves.png
    reports/figuras/14_confusion_matrix.png
    reports/figuras/15_threshold_analysis.png
    reports/logs/05_modelado.log
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import (
    FIGURES_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    TARGET,
    DataValidationError,
    configure_logging,
    save_json,
)

try:
    import psutil
except ImportError:  # pragma: no cover - psutil esta en requirements.txt
    psutil = None


MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
THRESHOLDS = np.unique(
    np.concatenate(
        [
            np.arange(0.01, 0.991, 0.005),
            np.array([0.5]),
        ]
    )
)

# Costos operativos por turno, en unidades relativas.
# FP: se marca como riesgo a alguien que iba a asistir -> costo de intervencion.
# FN: se marca como seguro a alguien que no asiste -> costo de cupo perdido.
COSTOS = {
    "base": {
        "descripcion": "Escenario base: recordatorio barato, cupo perdido costoso",
        "costo_fp": 0.25,
        "costo_fn": 1.00,
    },
    "conservador": {
        "descripcion": "Intervencion mas costosa o menos recursos disponibles",
        "costo_fp": 0.50,
        "costo_fn": 1.00,
    },
    "agresivo": {
        "descripcion": "Intervencion muy barata o alta preocupacion por el cupo",
        "costo_fp": 0.10,
        "costo_fn": 1.00,
    },
}


class RSSMonitor:
    """Mide el pico de memoria residente del proceso durante una operacion."""

    def __init__(self, interval: float = 0.01) -> None:
        self.interval = interval
        self.process = psutil.Process() if psutil is not None else None
        self.baseline = 0
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.process is None:
            return
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        assert self.process is not None
        while not self._stop.is_set():
            rss = self.process.memory_info().rss
            if rss > self.peak:
                self.peak = rss
            time.sleep(self.interval)

    def stop(self) -> dict[str, float]:
        if self.process is None:
            return {"rss_peak_mb": 0.0, "rss_delta_mb": 0.0}
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return {
            "rss_peak_mb": round(self.peak / 1024**2, 2),
            "rss_delta_mb": round((self.peak - self.baseline) / 1024**2, 2),
        }


def cargar_especificacion_features() -> tuple[list[str], list[str]]:
    """Lee la lista de features generada en la etapa 03."""
    path = REPORTS_DIR / "feature_list.json"
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "No existe reports/feature_list.json. "
            "Ejecuta primero python src/03_features.py."
        ) from exc

    numeric = list(spec.get("numeric_features", []))
    categorical = list(spec.get("categorical_features", []))
    if not numeric or not categorical:
        raise DataValidationError(
            "La especificacion de features esta vacia o incompleta."
        )
    return numeric, categorical


def cargar_split(nombre: str) -> pd.DataFrame:
    """Carga un split desde Parquet y usa CSV como respaldo."""
    parquet = PROCESSED_DIR / f"{nombre}.parquet"
    csv = PROCESSED_DIR / f"{nombre}.csv"

    try:
        if parquet.exists():
            return pd.read_parquet(parquet)
        if csv.exists():
            return pd.read_csv(csv)
        raise FileNotFoundError(
            f"No existe {parquet.name} ni {csv.name}. "
            "Ejecuta primero python src/03_features.py."
        )
    except (FileNotFoundError, pd.errors.ParserError, ValueError) as exc:
        raise DataValidationError(
            f"No se pudo cargar el split '{nombre}': {exc}"
        ) from exc


def construir_preprocesador(
    numericas: list[str],
    categoricas: list[str],
) -> ColumnTransformer:
    """Construye imputacion, escalado y codificacion one-hot."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    min_frequency=20,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numericas),
            ("cat", categorical_pipe, categoricas),
        ],
        remainder="drop",
    )


def construir_modelos() -> dict[str, Any]:
    """Devuelve los cuatro modelos que se compararan."""
    return {
        "dummy": DummyClassifier(strategy="prior", random_state=RANDOM_STATE),
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def metricas_por_umbral(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    umbral: float,
    costo_fp: float = 0.0,
    costo_fn: float = 0.0,
) -> dict[str, Any]:
    """Calcula metricas de clasificacion para un umbral concreto."""
    y_true_array = np.asarray(y_true).astype(int)
    y_pred = (y_score >= umbral).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_array, y_pred, labels=[0, 1]
    ).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    costo_esperado = fp * costo_fp + fn * costo_fn

    return {
        "threshold": round(float(umbral), 4),
        "accuracy": round(float(accuracy_score(y_true_array, y_pred)), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "specificity": round(float(specificity), 6),
        "f1": round(float(f1), 6),
        "tasa_predicha_positiva": round(float(y_pred.mean()), 6),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "costo_esperado_total": round(float(costo_esperado), 4),
        "costo_esperado_por_turno": round(
            float(costo_esperado / len(y_true_array)), 6
        ),
    }


def metricas_sin_umbral(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """Calcula metricas que no dependen de un umbral."""
    y_true_array = np.asarray(y_true).astype(int)
    score_clip = np.clip(y_score, 1e-7, 1 - 1e-7)
    return {
        "roc_auc": round(float(roc_auc_score(y_true_array, y_score)), 6),
        "pr_auc": round(float(average_precision_score(y_true_array, y_score)), 6),
        "brier": round(float(brier_score_loss(y_true_array, y_score)), 6),
        "log_loss": round(float(log_loss(y_true_array, score_clip)), 6),
    }


def analizar_umbrales(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    nombre_modelo: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evalua una grilla de umbrales para los tres escenarios de costo."""
    mejores: list[dict[str, Any]] = []
    grilla_base: list[dict[str, Any]] = []

    for escenario, costos in COSTOS.items():
        mejor: dict[str, Any] | None = None
        for umbral in THRESHOLDS:
            metricas = metricas_por_umbral(
                y_true,
                y_score,
                float(umbral),
                costo_fp=costos["costo_fp"],
                costo_fn=costos["costo_fn"],
            )
            fila = {
                "modelo": nombre_modelo,
                "escenario": escenario,
                **metricas,
                "umbral_teorico": round(
                    costos["costo_fp"]
                    / (costos["costo_fp"] + costos["costo_fn"]),
                    4,
                ),
            }
            if escenario == "base":
                grilla_base.append(fila)

            clave = (
                metricas["costo_esperado_total"],
                -metricas["f1"],
                abs(float(umbral) - 0.5),
            )
            if mejor is None or clave < (
                mejor["costo_esperado_total"],
                -mejor["f1"],
                abs(mejor["threshold"] - 0.5),
            ):
                mejor = fila

        if mejor is not None:
            mejores.append(mejor)

    return mejores, grilla_base


def medir_inferencia_individual(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    repeticiones: int = 100,
) -> float:
    """Mide la latencia mediana de predecir un unico turno (ms)."""
    muestra = X_test.iloc[[0]]
    pipeline.predict_proba(muestra)  # calentamiento
    tiempos = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        pipeline.predict_proba(muestra)
        tiempos.append(time.perf_counter() - inicio)
    return round(float(np.median(tiempos) * 1000), 4)


def entrenar_y_evaluar(
    modelos: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    numericas: list[str],
    categoricas: list[str],
    logger,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Entrena cada modelo, mide recursos y evalua en validation."""
    resultados: dict[str, Any] = {}
    filas_comparacion: list[dict[str, Any]] = []
    filas_umbral: list[dict[str, Any]] = []
    filas_grilla: list[dict[str, Any]] = []

    for nombre, estimador in modelos.items():
        logger.info("-" * 68)
        logger.info("Entrenando %s", nombre)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", construir_preprocesador(numericas, categoricas)),
                ("model", estimador),
            ]
        )

        monitor = RSSMonitor()
        monitor.start()
        inicio_fit = time.perf_counter()
        try:
            pipeline.fit(X_train, y_train)
        finally:
            tiempo_fit = time.perf_counter() - inicio_fit
            memoria = monitor.stop()

        inicio_pred = time.perf_counter()
        y_score_val = pipeline.predict_proba(X_val)[:, 1]
        tiempo_pred = time.perf_counter() - inicio_pred

        inicio_transform = time.perf_counter()
        pipeline.named_steps["preprocessor"].transform(X_val)
        tiempo_transform = time.perf_counter() - inicio_transform

        latencia_ms = medir_inferencia_individual(pipeline, X_test)
        metricas_globales = metricas_sin_umbral(y_val, y_score_val)
        metricas_05 = metricas_por_umbral(
            y_val,
            y_score_val,
            0.5,
            costo_fp=COSTOS["base"]["costo_fp"],
            costo_fn=COSTOS["base"]["costo_fn"],
        )
        mejores, grilla_base = analizar_umbrales(y_val, y_score_val, nombre)

        resultado_modelo = {
            "modelo": nombre,
            "validation": {
                **metricas_globales,
                "at_0.5": metricas_05,
            },
            "thresholds": {
                fila["escenario"]: fila for fila in mejores
            },
            "tiempo": {
                "fit_s": round(float(tiempo_fit), 4),
                "predict_s": round(float(tiempo_pred), 4),
                "transform_s": round(float(tiempo_transform), 4),
                "single_row_ms": latencia_ms,
                **memoria,
            },
            "y_score_validation": y_score_val,
        }
        resultados[nombre] = resultado_modelo
        filas_umbral.extend(mejores)
        filas_grilla.extend(grilla_base)

        # Se guarda el pipeline completo (preprocesamiento + modelo).
        # Random Forest se comprime porque su artefacto supera los 100 MB
        # sin compresion y no debe romper el limite de GitHub.
        ruta_modelo = MODELS_DIR / f"{nombre}.joblib"
        joblib.dump(
            pipeline,
            ruta_modelo,
            compress=3 if nombre == "random_forest" else 0,
        )
        tamano_mb = round(ruta_modelo.stat().st_size / 1024**2, 2)

        fila_comparacion = {
            "modelo": nombre,
            "fit_s": resultado_modelo["tiempo"]["fit_s"],
            "predict_s": resultado_modelo["tiempo"]["predict_s"],
            "single_row_ms": latencia_ms,
            "rss_peak_mb": memoria["rss_peak_mb"],
            "rss_delta_mb": memoria["rss_delta_mb"],
            "artifact_size_mb": tamano_mb,
            "validation_roc_auc": metricas_globales["roc_auc"],
            "validation_pr_auc": metricas_globales["pr_auc"],
            "validation_brier": metricas_globales["brier"],
            "validation_f1_at_0.5": metricas_05["f1"],
            "validation_precision_at_0.5": metricas_05["precision"],
            "validation_recall_at_0.5": metricas_05["recall"],
            "validation_threshold_base": resultado_modelo["thresholds"]["base"][
                "threshold"
            ],
            "validation_f1_at_base": resultado_modelo["thresholds"]["base"]["f1"],
            "validation_precision_at_base": resultado_modelo["thresholds"]["base"][
                "precision"
            ],
            "validation_recall_at_base": resultado_modelo["thresholds"]["base"][
                "recall"
            ],
            "validation_cost_base": resultado_modelo["thresholds"]["base"][
                "costo_esperado_por_turno"
            ],
        }
        filas_comparacion.append(fila_comparacion)

        logger.info(
            "  fit=%.2fs | predict=%.3fs | 1 fila=%.3fms | "
            "RSS pico=%.1f MB | artefacto=%.2f MB",
            tiempo_fit,
            tiempo_pred,
            latencia_ms,
            memoria["rss_peak_mb"],
            tamano_mb,
        )
        logger.info(
            "  ROC-AUC=%.4f | PR-AUC=%.4f | F1@0.5=%.4f | "
            "umbral base=%.3f | F1@umbral=%.4f | costo/turno=%.4f",
            metricas_globales["roc_auc"],
            metricas_globales["pr_auc"],
            metricas_05["f1"],
            resultado_modelo["thresholds"]["base"]["threshold"],
            resultado_modelo["thresholds"]["base"]["f1"],
            resultado_modelo["thresholds"]["base"]["costo_esperado_por_turno"],
        )

    return (
        resultados,
        pd.DataFrame(filas_comparacion),
        pd.DataFrame(filas_umbral),
        pd.DataFrame(filas_grilla),
    )


def guardar_figuras(
    resultados: dict[str, Any],
    comparacion: pd.DataFrame,
    mejor_modelo: str,
    y_val: pd.Series,
    y_test: pd.Series,
    test_metricas: dict[str, Any],
    logger,
) -> None:
    """Genera las figuras de evaluacion del modelado."""
    sns.set_theme(style="whitegrid", palette="Set2")

    # 1. Comparacion de metricas principales.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    comparacion_plot = comparacion.set_index("modelo")[
        ["validation_pr_auc", "validation_roc_auc", "validation_f1_at_base"]
    ].rename(
        columns={
            "validation_pr_auc": "PR-AUC",
            "validation_roc_auc": "ROC-AUC",
            "validation_f1_at_base": "F1 en umbral base",
        }
    )
    comparacion_plot.plot(kind="bar", ax=ax)
    ax.set_title("Comparacion de modelos en validation")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Metrica")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "11_model_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Curvas ROC.
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for nombre, resultado in resultados.items():
        fpr, tpr, _ = roc_curve(y_val, resultado["y_score_validation"])
        ax.plot(
            fpr,
            tpr,
            label=f"{nombre} (AUC={resultado['validation']['roc_auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
    ax.set_title("Curvas ROC en validation")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "12_roc_curves.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Curvas precision-recall.
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for nombre, resultado in resultados.items():
        precision, recall, _ = precision_recall_curve(
            y_val, resultado["y_score_validation"]
        )
        ax.plot(
            recall,
            precision,
            label=f"{nombre} (AP={resultado['validation']['pr_auc']:.3f})",
        )
    ax.axhline(
        y_val.mean(),
        linestyle="--",
        color="gray",
        label=f"Prevalencia={y_val.mean():.3f}",
    )
    ax.set_title("Curvas precision-recall en validation")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "13_pr_curves.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Matriz de confusion del modelo final en test.
    cm = np.array(
        [
            [test_metricas["tn"], test_metricas["fp"]],
            [test_metricas["fn"], test_metricas["tp"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred. asiste", "Pred. no asiste"],
        yticklabels=["Asistio", "No asistio"],
        ax=ax,
    )
    ax.set_title(
        f"Matriz de confusion - {mejor_modelo}\n"
        f"test, umbral={test_metricas['threshold']:.3f}"
    )
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Realidad")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "14_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    # 5. Umbral vs metricas y costo esperado para el modelo final.
    mejor = resultados[mejor_modelo]
    grid = pd.DataFrame(
        [
            metricas_por_umbral(
                y_val,
                mejor["y_score_validation"],
                float(umbral),
                costo_fp=COSTOS["base"]["costo_fp"],
                costo_fn=COSTOS["base"]["costo_fn"],
            )
            for umbral in THRESHOLDS
        ]
    )
    umbral_final = mejor["thresholds"]["base"]["threshold"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(grid["threshold"], grid["precision"], label="Precision")
    axes[0].plot(grid["threshold"], grid["recall"], label="Recall")
    axes[0].plot(grid["threshold"], grid["f1"], label="F1")
    axes[0].axvline(
        umbral_final,
        color="black",
        linestyle="--",
        label=f"Umbral elegido={umbral_final:.3f}",
    )
    axes[0].set_title("Metricas segun umbral")
    axes[0].set_xlabel("Umbral")
    axes[0].set_ylabel("Metrica")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)

    axes[1].plot(
        grid["threshold"],
        grid["costo_esperado_por_turno"],
        color="#8B3A3A",
        label="Costo esperado por turno",
    )
    axes[1].axvline(
        umbral_final,
        color="black",
        linestyle="--",
        label=f"Umbral elegido={umbral_final:.3f}",
    )
    axes[1].set_title("Costo esperado segun umbral (escenario base)")
    axes[1].set_xlabel("Umbral")
    axes[1].set_ylabel("Costo relativo por turno")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "15_threshold_analysis.png", bbox_inches="tight")
    plt.close(fig)

    logger.info("Figuras de modelado guardadas en %s", FIGURES_DIR)


def escribir_model_card(
    mejor_modelo: str,
    comparacion: pd.DataFrame,
    test_metricas: dict[str, Any],
    test_globales: dict[str, float],
    umbral: float,
) -> None:
    """Genera una ficha Markdown con el resumen del modelo final."""
    filas = []
    for _, fila in comparacion.iterrows():
        filas.append(
            "| {modelo} | {fit_s:.2f} | {rss_delta_mb:.1f} | "
            "{artifact_size_mb:.2f} | {validation_pr_auc:.4f} | "
            "{validation_roc_auc:.4f} | {validation_f1_at_base:.4f} |".format(
                **fila
            )
        )

    mejor_global = comparacion.sort_values(
        "validation_pr_auc", ascending=False
    ).iloc[0]
    nota_seleccion = ""
    if mejor_global["modelo"] != mejor_modelo:
        nota_seleccion = (
            f"\n> **Nota de despliegue:** `{mejor_global['modelo']}` obtuvo "
            f"el mejor PR-AUC de validation ({mejor_global['validation_pr_auc']:.4f}), "
            "pero su artefacto es mucho mas pesado. Para la app se eligio "
            f"`{mejor_modelo}`, con rendimiento cercano y bajo costo de "
            "almacenamiento e inferencia.\n"
        )

    contenido = f"""# Ficha del modelo

## Modelo final

- **Modelo:** `{mejor_modelo}`
- **Umbral operativo:** `{umbral:.4f}`
- **Escenario de costo:** base (`C_FP=0.25`, `C_FN=1.00`)
- **Target:** `{TARGET}` (1 = no asistio)
{nota_seleccion}
## Metricas en test

| Metrica | Valor |
|---|---:|
| ROC-AUC | {test_globales['roc_auc']:.4f} |
| PR-AUC | {test_globales['pr_auc']:.4f} |
| Brier | {test_globales['brier']:.4f} |
| Accuracy | {test_metricas['accuracy']:.4f} |
| Precision | {test_metricas['precision']:.4f} |
| Recall | {test_metricas['recall']:.4f} |
| F1 | {test_metricas['f1']:.4f} |
| Especificidad | {test_metricas['specificity']:.4f} |
| Tasa predicha positiva | {test_metricas['tasa_predicha_positiva']:.4f} |

## Comparacion en validation

| Modelo | Fit (s) | RAM delta (MB) | Artefacto (MB) | PR-AUC | ROC-AUC | F1 umbral base |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(filas)}

## Advertencia

El modelo es un prototipo academico entrenado con datos de Brasil (2016).
No debe usarse para negar, retrasar o priorizar atención clínica sin
validacion local, auditoria de sesgos y supervision humana.
"""
    (REPORTS_DIR / "model_card.md").write_text(contenido, encoding="utf-8")


def main() -> int:
    logger = configure_logging("05_modelado")
    logger.info("=" * 68)
    logger.info("ETAPA 6: MODELADO Y SELECCION DE UMBRAL")
    logger.info("=" * 68)

    try:
        numericas, categoricas = cargar_especificacion_features()
        train = cargar_split("train")
        validation = cargar_split("validation")
        test = cargar_split("test")

        features = numericas + categoricas
        faltantes = sorted(set(features + [TARGET]) - set(train.columns))
        if faltantes:
            raise DataValidationError(
                "Faltan columnas en train: " + ", ".join(faltantes)
            )

        X_train = train[features].copy()
        y_train = train[TARGET].astype(int)
        X_val = validation[features].copy()
        y_val = validation[TARGET].astype(int)
        X_test = test[features].copy()
        y_test = test[TARGET].astype(int)

        logger.info(
            "Train=%s | Validation=%s | Test=%s | Features=%s",
            f"{len(X_train):,}",
            f"{len(X_val):,}",
            f"{len(X_test):,}",
            len(features),
        )
        logger.info(
            "Tasa no-show: train=%.2f%% | validation=%.2f%% | test=%.2f%%",
            y_train.mean() * 100,
            y_val.mean() * 100,
            y_test.mean() * 100,
        )

        modelos = construir_modelos()
        resultados, comparacion, umbrales, grilla = entrenar_y_evaluar(
            modelos,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            numericas,
            categoricas,
            logger,
        )

        # El modelo se elige por PR-AUC en validation: mide la capacidad de
        # ordenar riesgo sin depender de un umbral y es adecuada con desbalance.
        # Para que la app sea liviana y rapida, se excluyen artefactos > 10 MB.
        # Random Forest obtiene el mejor PR-AUC, pero su serializacion pesa
        # decenas de MB y tarda mas en inferencia; HGB queda como final.
        tamano_maximo_mb = 10.0
        candidatos = [
            nombre
            for nombre in resultados
            if comparacion.loc[
                comparacion["modelo"] == nombre, "artifact_size_mb"
            ].iloc[0]
            <= tamano_maximo_mb
        ]
        mejor_modelo = max(
            candidatos,
            key=lambda nombre: resultados[nombre]["validation"]["pr_auc"],
        )
        mejor_resultado = resultados[mejor_modelo]
        umbral_final = mejor_resultado["thresholds"]["base"]["threshold"]
        logger.info(
            "Modelo elegido para despliegue: %s "
            "(PR-AUC=%.4f, umbral=%.4f, artefacto<=%.0f MB)",
            mejor_modelo,
            mejor_resultado["validation"]["pr_auc"],
            umbral_final,
            tamano_maximo_mb,
        )
        mejor_global = max(
            resultados,
            key=lambda nombre: resultados[nombre]["validation"]["pr_auc"],
        )
        if mejor_global != mejor_modelo:
            logger.info(
                "Mejor PR-AUC absoluto: %s (%.4f); se prioriza un modelo "
                "desplegable de bajo peso.",
                mejor_global,
                resultados[mejor_global]["validation"]["pr_auc"],
            )

        # El test se toca una sola vez, para el modelo ya elegido.
        pipeline_final = joblib.load(MODELS_DIR / f"{mejor_modelo}.joblib")
        y_score_test = pipeline_final.predict_proba(X_test)[:, 1]
        test_globales = metricas_sin_umbral(y_test, y_score_test)
        test_default = metricas_por_umbral(
            y_test,
            y_score_test,
            0.5,
            costo_fp=COSTOS["base"]["costo_fp"],
            costo_fn=COSTOS["base"]["costo_fn"],
        )
        test_final = metricas_por_umbral(
            y_test,
            y_score_test,
            umbral_final,
            costo_fp=COSTOS["base"]["costo_fp"],
            costo_fn=COSTOS["base"]["costo_fn"],
        )

        comparacion.to_csv(
            REPORTS_DIR / "model_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )
        umbrales.to_csv(
            REPORTS_DIR / "threshold_analysis.csv",
            index=False,
            encoding="utf-8-sig",
        )
        grilla.to_csv(
            REPORTS_DIR / "threshold_grid.csv",
            index=False,
            encoding="utf-8-sig",
        )

        metricas_json = {
            "metadata": {
                "random_state": RANDOM_STATE,
                "target": TARGET,
                "train_rows": int(len(X_train)),
                "validation_rows": int(len(X_val)),
                "test_rows": int(len(X_test)),
                "features_numericas": numericas,
                "features_categoricas": categoricas,
                "modelos": list(modelos),
                "criterio_seleccion": (
                    "mayor PR-AUC en validation entre modelos con artefacto "
                    "<= 10 MB, para priorizar un despliegue liviano"
                ),
            },
            "costos": COSTOS,
            "comparacion_validation": comparacion.to_dict(orient="records"),
            "modelos": {
                nombre: {
                    clave: valor
                    for clave, valor in resultado.items()
                    if clave != "y_score_validation"
                }
                for nombre, resultado in resultados.items()
            },
            "modelo_final": {
                "nombre": mejor_modelo,
                "umbral": umbral_final,
                "validation": {
                    clave: valor
                    for clave, valor in mejor_resultado["validation"].items()
                },
                "test": {
                    **test_globales,
                    "at_0.5": test_default,
                    "at_selected": test_final,
                },
            },
        }
        save_json(REPORTS_DIR / "model_metrics.json", metricas_json)

        metadata_app = {
            "model_name": mejor_modelo,
            "target": TARGET,
            "threshold": umbral_final,
            "scenario": "base",
            "cost_fp": COSTOS["base"]["costo_fp"],
            "cost_fn": COSTOS["base"]["costo_fn"],
            "features_numeric": numericas,
            "features_categorical": categoricas,
            "validation": mejor_resultado["validation"],
            "validation_at_selected": mejor_resultado["thresholds"]["base"],
            "test": {
                **test_globales,
                "at_0.5": test_default,
                "at_selected": test_final,
            },
            "trained_rows": int(len(X_train)),
            "advertencia": (
                "Prototipo academico con datos de Brasil (2016). "
                "No usar para decisiones clinicas sin validacion local."
            ),
        }
        save_json(MODELS_DIR / "model_metadata.json", metadata_app)
        joblib.dump(pipeline_final, MODELS_DIR / "final_model.joblib")

        guardar_figuras(
            resultados,
            comparacion,
            mejor_modelo,
            y_val,
            y_test,
            test_final,
            logger,
        )
        escribir_model_card(
            mejor_modelo,
            comparacion,
            test_final,
            test_globales,
            umbral_final,
        )

        logger.info("-" * 68)
        logger.info("Resultado final en test (%s, umbral=%.4f):", mejor_modelo, umbral_final)
        logger.info(
            "  ROC-AUC=%.4f | PR-AUC=%.4f | accuracy=%.4f | precision=%.4f | "
            "recall=%.4f | F1=%.4f",
            test_globales["roc_auc"],
            test_globales["pr_auc"],
            test_final["accuracy"],
            test_final["precision"],
            test_final["recall"],
            test_final["f1"],
        )
        logger.info(
            "  Matriz de confusion: TN=%s | FP=%s | FN=%s | TP=%s",
            test_final["tn"],
            test_final["fp"],
            test_final["fn"],
            test_final["tp"],
        )
        logger.info(
            "  Costo esperado base: %.4f por turno",
            test_final["costo_esperado_por_turno"],
        )
        logger.info("Modelo final guardado en models/final_model.joblib")
        logger.info("La etapa 6 finalizo correctamente.")

    except (
        FileNotFoundError,
        DataValidationError,
        pd.errors.ParserError,
        ValueError,
    ) as exc:
        logger.error("No se pudo completar el modelado: %s", exc)
        return 1
    except MemoryError:
        logger.error("Memoria insuficiente durante el modelado.")
        return 1
    else:
        return 0
    finally:
        logger.info("Fin del script 05_modelado.py")


if __name__ == "__main__":
    raise SystemExit(main())
