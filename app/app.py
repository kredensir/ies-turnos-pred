"""
Aplicacion Streamlit del Proyecto Integrador.

Paginas:
    1. Panorama: indicadores y visualizaciones del dataset.
    2. Prediccion: formulario individual y probabilidad estimada de no-show.
    3. Modelo: metricas, umbral operativo y advertencias.

La app carga el modelo final desde models/final_model.joblib y los metadatos
desde models/model_metadata.json. Si el modelo no existe, la pagina de
prediccion muestra una advertencia en lugar de fallar.
"""

from __future__ import annotations

import json
from datetime import date, time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
PARQUET_PATH = PROCESSED_DIR / "turnos_clean.parquet"
CSV_PATH = PROCESSED_DIR / "turnos_clean.csv"
MODEL_PATH = MODELS_DIR / "final_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
MODEL_CARD_PATH = ROOT / "reports" / "model_card.md"

st.set_page_config(
    page_title="Prediccion de inasistencias a turnos medicos",
    page_icon="🩺",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cargar_datos() -> pd.DataFrame:
    """Carga el dataset limpio desde Parquet o CSV."""
    if PARQUET_PATH.exists():
        return pd.read_parquet(PARQUET_PATH)
    if CSV_PATH.exists():
        return pd.read_csv(
            CSV_PATH,
            parse_dates=["ScheduledDay", "AppointmentDay"],
        )
    raise FileNotFoundError(
        "No se encontro el dataset limpio. "
        "Ejecuta src/02_limpieza.py antes de abrir la app."
    )


@st.cache_resource(show_spinner=False)
def cargar_modelo():
    """Carga el pipeline final y sus metadatos."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    modelo = joblib.load(MODEL_PATH)
    metadatos = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return modelo, metadatos


def clasificar_demora(dias: int) -> str:
    """Replica la categorizacion usada en la etapa de limpieza."""
    if dias <= 0:
        return "0"
    if dias <= 7:
        return "1-7"
    if dias <= 14:
        return "8-14"
    if dias <= 30:
        return "15-30"
    return ">30"


def clasificar_edad(edad: int) -> str:
    """Replica la categorizacion usada en la etapa de limpieza."""
    if edad <= 12:
        return "0-12"
    if edad <= 18:
        return "13-18"
    if edad <= 30:
        return "19-30"
    if edad <= 45:
        return "31-45"
    if edad <= 60:
        return "46-60"
    return "61+"


def construir_fila_prediccion(
    edad: int,
    genero: str,
    barrio: str,
    fecha_reserva: date,
    hora_reserva: time,
    fecha_cita: date,
    scholarship: int,
    hipertension: int,
    diabetes: int,
    alcoholism: int,
    handcap: int,
    sms: int,
) -> tuple[pd.DataFrame, int]:
    """Construye una fila con el mismo esquema usado en el entrenamiento."""
    reserva = pd.Timestamp(fecha_reserva)
    cita = pd.Timestamp(fecha_cita)
    dias = (cita.normalize() - reserva.normalize()).days
    corregida = int(dias < 0)
    dias = max(dias, 0)

    weekday_reserva = int(reserva.dayofweek)
    weekday_cita = int(cita.dayofweek)
    chronic = int(hipertension + diabetes + alcoholism)

    fila = {
        "Age": int(edad),
        "waiting_days": int(dias),
        "scheduled_hour": int(hora_reserva.hour),
        "scheduled_weekday": weekday_reserva,
        "appointment_weekday": weekday_cita,
        "appointment_week": int(cita.isocalendar().week),
        "appointment_month": int(cita.month),
        "is_weekend": int(weekday_cita >= 5),
        "chronic_conditions": chronic,
        "has_chronic_condition": int(chronic > 0),
        "handicap_any": int(handcap > 0),
        "Scholarship": int(scholarship),
        "Hipertension": int(hipertension),
        "Diabetes": int(diabetes),
        "Alcoholism": int(alcoholism),
        "Handcap": int(handcap),
        "SMS_received": int(sms),
        "is_same_day": int(dias == 0),
        "age_imputed": 0,
        "age_capped": 0,
        "waiting_days_corrected": corregida,
        "Gender": genero,
        "Neighbourhood": barrio,
        "age_group": clasificar_edad(edad),
        "waiting_days_bucket": clasificar_demora(dias),
    }
    return pd.DataFrame([fila]), dias


def mostrar_panorama(df: pd.DataFrame) -> None:
    """Indicadores y graficos descriptivos del dataset."""
    total = len(df)
    tasa_no_show = df["no_show"].mean() if "no_show" in df.columns else float("nan")
    pacientes = df["PatientId"].nunique() if "PatientId" in df.columns else 0
    barrios = df["Neighbourhood"].nunique() if "Neighbourhood" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Turnos", f"{total:,}".replace(",", "."))
    col2.metric("Tasa de no-show", f"{tasa_no_show:.2%}")
    col3.metric("Pacientes unicos", f"{pacientes:,}".replace(",", "."))
    col4.metric("Barrios", f"{barrios:,}".replace(",", "."))

    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown("**No-show segun demora entre reserva y cita**")
        if {"waiting_days_bucket", "no_show"}.issubset(df.columns):
            orden = ["0", "1-7", "8-14", "15-30", ">30"]
            tabla = (
                df.groupby("waiting_days_bucket", observed=True)["no_show"]
                .agg(["count", "mean"])
                .reindex(orden)
                .reset_index()
            )
            tabla["tasa_porcentaje"] = tabla["mean"] * 100
            st.bar_chart(
                tabla,
                x="waiting_days_bucket",
                y="tasa_porcentaje",
                x_label="Dias de espera",
                y_label="No-show (%)",
            )
            st.caption(
                "Los turnos del mismo dia tienen una tasa de ausencia mucho menor "
                "que los turnos agendados con varios dias de anticipacion."
            )
        else:
            st.info("Faltan columnas para construir esta visualizacion.")

    with right:
        st.markdown("**No-show segun recepcion de SMS**")
        if {"SMS_received", "no_show"}.issubset(df.columns):
            tabla_sms = (
                df.groupby("SMS_received", observed=True)["no_show"]
                .agg(["count", "mean"])
                .reset_index()
            )
            tabla_sms["grupo"] = tabla_sms["SMS_received"].map(
                {0: "Sin SMS", 1: "Con SMS"}
            )
            tabla_sms["tasa_porcentaje"] = tabla_sms["mean"] * 100
            st.bar_chart(
                tabla_sms,
                x="grupo",
                y="tasa_porcentaje",
                x_label="Grupo",
                y_label="No-show (%)",
            )
            st.caption(
                "La diferencia es descriptiva, no causal: el SMS pudo haberse "
                "enviado a pacientes de mayor riesgo."
            )
        else:
            st.info("Faltan columnas para construir esta visualizacion.")

    st.divider()
    st.subheader("Hallazgos del EDA")
    st.markdown(
        """
        - **Desbalance:** la clase positiva es aproximadamente el 20 %; el accuracy
          por si solo no alcanza.
        - **Demora:** es el patron operativo mas fuerte. Los turnos del mismo dia
          tienen una tasa mucho menor.
        - **Edad:** la relacion no es lineal; adolescentes y adultos jovenes
          concentran mas ausentismo.
        - **Barrio:** hay variacion geografica, pero puede reflejar diferencias
          socioeconomicas y de acceso. No debe usarse para estigmatizar.
        - **SMS:** el grupo con SMS tiene mas no-show; se interpreta como
          asociacion, no como causalidad.
        """
    )


def mostrar_prediccion(
    df: pd.DataFrame,
    modelo,
    metadatos: dict | None,
) -> None:
    """Formulario de prediccion individual."""
    st.subheader("Prediccion individual")
    st.caption(
        "Completa los datos del turno. El modelo devuelve una probabilidad "
        "estimada de inasistencia; no es una decision automatica."
    )

    if modelo is None or metadatos is None:
        st.warning(
            "Todavia no existe el modelo final. Ejecuta primero "
            "`python src/05_modelado.py`."
        )
        return

    barrios = sorted(df["Neighbourhood"].dropna().astype(str).unique().tolist())
    barrio_default = "JARDIM DA PENHA" if "JARDIM DA PENHA" in barrios else barrios[0]

    with st.form("formulario_prediccion"):
        st.markdown("**Datos del paciente y del turno**")
        c1, c2, c3 = st.columns(3)
        edad = c1.number_input("Edad", min_value=0, max_value=100, value=30, step=1)
        genero = c2.selectbox("Genero registrado", options=["F", "M"])
        barrio = c3.selectbox("Barrio", options=barrios, index=barrios.index(barrio_default))

        c4, c5, c6 = st.columns(3)
        fecha_reserva = c4.date_input("Fecha de reserva", value=date(2016, 5, 2))
        hora_reserva = c5.time_input("Hora de reserva", value=time(9, 0))
        fecha_cita = c6.date_input("Fecha de la cita", value=date(2016, 5, 9))

        st.markdown("**Condiciones declaradas**")
        c7, c8, c9, c10 = st.columns(4)
        scholarship = c7.checkbox("Beca social")
        hipertension = c8.checkbox("Hipertension")
        diabetes = c9.checkbox("Diabetes")
        alcoholism = c10.checkbox("Alcoholismo")

        c11, c12 = st.columns(2)
        handcap = c11.selectbox("Nivel de discapacidad", options=[0, 1, 2, 3, 4], index=0)
        sms = c12.checkbox("Recibio SMS")

        enviado = st.form_submit_button("Estimar riesgo", type="primary")

    if not enviado:
        st.info("Completa el formulario y presiona **Estimar riesgo**.")
        return

    fila, dias = construir_fila_prediccion(
        edad=int(edad),
        genero=genero,
        barrio=barrio,
        fecha_reserva=fecha_reserva,
        hora_reserva=hora_reserva,
        fecha_cita=fecha_cita,
        scholarship=int(scholarship),
        hipertension=int(hipertension),
        diabetes=int(diabetes),
        alcoholism=int(alcoholism),
        handcap=int(handcap),
        sms=int(sms),
    )

    if fecha_cita < fecha_reserva:
        st.warning(
            "La fecha de la cita es anterior a la reserva. "
            "Para el calculo se corrigio la demora a 0 dias."
        )

    columnas = metadatos["features_numeric"] + metadatos["features_categorical"]
    try:
        X = fila[columnas]
        probabilidad = float(modelo.predict_proba(X)[0, 1])
    except (KeyError, ValueError, IndexError) as exc:
        st.error(f"No se pudo calcular la prediccion: {exc}")
        return

    umbral = float(metadatos["threshold"])
    riesgo_alto = probabilidad >= umbral

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilidad de no-show", f"{probabilidad:.1%}")
    c2.metric("Umbral operativo", f"{umbral:.1%}")
    c3.metric("Demora calculada", f"{dias} dias")
    st.progress(min(max(probabilidad, 0.0), 1.0))

    if riesgo_alto:
        st.error(
            "**Riesgo alto de inasistencia.** Se recomienda priorizar un "
            "recordatorio personalizado o una llamada de confirmacion. "
            "La decision final debe quedar en manos del equipo de salud."
        )
    else:
        st.success(
            "**Riesgo bajo de inasistencia.** Se puede mantener la gestion "
            "habitual del turno, sin acciones adicionales automaticas."
        )

    with st.expander("Detalle de la entrada utilizada"):
        detalle = fila.T.rename(columns={0: "valor"}).astype(str)
        st.dataframe(detalle, width="stretch")

    st.caption(
        "Esta probabilidad es una estimacion estadistica. No debe usarse para "
        "negar, retrasar ni priorizar atencion clinica sin supervision humana."
    )


def mostrar_modelo(metadatos: dict | None) -> None:
    """Ficha tecnica del modelo final."""
    st.subheader("Modelo y metricas")
    if metadatos is None:
        st.warning(
            "Todavia no existe models/model_metadata.json. "
            "Ejecuta `python src/05_modelado.py`."
        )
        return

    test = metadatos["test"]
    seleccionado = test["at_selected"]
    st.markdown(
        f"**Modelo final:** `{metadatos['model_name']}`  \n"
        f"**Umbral operativo:** `{metadatos['threshold']:.2%}`  \n"
        f"**Criterio:** escenario base con costo FP = "
        f"`{metadatos['cost_fp']}` y costo FN = `{metadatos['cost_fn']}`."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC (test)", f"{test['roc_auc']:.3f}")
    c2.metric("PR-AUC (test)", f"{test['pr_auc']:.3f}")
    c3.metric("Recall (test)", f"{seleccionado['recall']:.1%}")
    c4.metric("F1 (test)", f"{seleccionado['f1']:.3f}")

    st.markdown(
        """
        **Como se lee:** el umbral se eligio en validation minimizando un costo
        esperado. Un falso negativo (predecir que asistira y que no asista)
        cuesta un cupo perdido; un falso positivo (predecir riesgo y que si
        asista) cuesta una intervencion innecesaria. Con el escenario base, se
        prioriza detectar la mayor cantidad posible de ausencias, aun a costa de
        mas falsos positivos.
        """
    )

    st.markdown(
        f"""
        **Metricas completas en test (umbral {seleccionado['threshold']:.2f}):**

        | Metrica | Valor |
        |---|---:|
        | Accuracy | {seleccionado['accuracy']:.4f} |
        | Precision | {seleccionado['precision']:.4f} |
        | Recall | {seleccionado['recall']:.4f} |
        | Especificidad | {seleccionado['specificity']:.4f} |
        | F1 | {seleccionado['f1']:.4f} |
        | Tasa predicha positiva | {seleccionado['tasa_predicha_positiva']:.4f} |
        | Brier | {test['brier']:.4f} |
        """
    )

    if MODEL_CARD_PATH.exists():
        st.download_button(
            "Descargar ficha del modelo",
            data=MODEL_CARD_PATH.read_text(encoding="utf-8"),
            file_name="model_card.md",
            mime="text/markdown",
        )


def main() -> None:
    st.title("Prediccion de inasistencias a turnos medicos")
    st.caption(
        "Proyecto Integrador - Ciencia de Datos + Programacion II - "
        "Prototipo academico"
    )
    st.warning(
        "Esta aplicacion es un prototipo academico. No reemplaza el criterio "
        "clinico ni decide sobre la atencion de una persona."
    )

    try:
        df = cargar_datos()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    modelo, metadatos = cargar_modelo()

    tab_panorama, tab_prediccion, tab_modelo = st.tabs(
        ["Panorama", "Prediccion", "Modelo"]
    )
    with tab_panorama:
        mostrar_panorama(df)
    with tab_prediccion:
        mostrar_prediccion(df, modelo, metadatos)
    with tab_modelo:
        mostrar_modelo(metadatos)

    with st.expander("Fuente, privacidad y limitaciones"):
        st.markdown(
            """
            **Fuente:** Medical Appointment No Shows (Kaggle), Brasil, 2016.

            **Privacidad:** la app no muestra identificadores de pacientes.

            **Limitaciones:** los datos son historicos y de otro contexto. El
            modelo no demuestra causalidad: que una variable se asocie con el
            no-show no significa que intervenir sobre ella lo reduzca. Antes de
            un uso real se requiere validacion local, auditoria de sesgos,
            calibracion y supervision humana.
            """
        )


if __name__ == "__main__":
    main()
