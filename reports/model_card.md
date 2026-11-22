# Ficha del modelo

## Modelo final

- **Modelo:** `hist_gradient_boosting`
- **Umbral operativo:** `0.4400`
- **Escenario de costo:** base (`C_FP=0.25`, `C_FN=1.00`)
- **Target:** `no_show` (1 = no asistio)

> **Nota de despliegue:** `random_forest` obtuvo el mejor PR-AUC de validation (0.3611), pero su artefacto es mucho mas pesado. Para la app se eligio `hist_gradient_boosting`, con rendimiento cercano y bajo costo de almacenamiento e inferencia.

## Metricas en test

| Metrica | Valor |
|---|---:|
| ROC-AUC | 0.7335 |
| PR-AUC | 0.3699 |
| Brier | 0.2076 |
| Accuracy | 0.5570 |
| Precision | 0.3024 |
| Recall | 0.8841 |
| F1 | 0.4507 |
| Especificidad | 0.4724 |
| Tasa predicha positiva | 0.6008 |

## Comparacion en validation

| Modelo | Fit (s) | RAM delta (MB) | Artefacto (MB) | PR-AUC | ROC-AUC | F1 umbral base |
|---|---:|---:|---:|---:|---:|---:|
| dummy | 1.81 | 140.4 | 0.01 | 0.1953 | 0.5000 | 0.0000 |
| logistic_regression | 6.98 | 133.7 | 0.01 | 0.3426 | 0.7240 | 0.4360 |
| random_forest | 77.62 | 235.4 | 41.82 | 0.3611 | 0.7334 | 0.4360 |
| hist_gradient_boosting | 76.54 | 154.1 | 0.92 | 0.3549 | 0.7300 | 0.4330 |

## Advertencia

El modelo es un prototipo academico entrenado con datos de Brasil (2016).
No debe usarse para negar, retrasar o priorizar atención clínica sin
validacion local, auditoria de sesgos y supervision humana.
