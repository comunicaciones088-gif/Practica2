import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn

st.set_page_config(
    page_title="Predicción de estudiantes aprobados o no aprobados ",
    layout="centered"
)

st.title("Predicción de estudiantes aprobados o no aprobados")

st.write(
    "Esta aplicación usa un modelo registrado en MLflow para predecir "
    "si un estudiante aprobará o no."
)

# Conexión a MLflow
mlflow.set_tracking_uri("http://127.0.0.1:9090")

# Cambia este nombre por el modelo que registraste en MLflow
MODEL_URI = "models:/Practica2/1"

@st.cache_resource
def cargar_modelo():
    return mlflow.sklearn.load_model(MODEL_URI)

model = cargar_modelo()

st.sidebar.header("Configuración")
st.sidebar.write(f"Modelo cargado: `{MODEL_URI}`")

st.subheader("Datos del estudiante")

carrera = st.selectbox(
    "Carrera",
    ["Computacion", "Derecho", "Economia", "Medicina", "Arquitectura", "Industrial"]
)

modalidad = st.selectbox(
    "Modalidad",
    ["Presencial", "Virtual", "Hibrida"]
)

beca = st.selectbox(
    "¿Tiene beca?",
    ["Si", "No"]
)

edad = st.number_input(
    "Edad",
    min_value=18,
    max_value=30,
    value=22
)

promedio = st.number_input(
    "Promedio",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.1
)

asistencias = st.number_input(
    "Asistencias",
    min_value=0,
    max_value=100,
    value=80
)

datos = pd.DataFrame([{
    "carrera": carrera,
    "modalidad": modalidad,
    "beca": beca,
    "edad": edad,
    "promedio": promedio,
    "asistencias": asistencias
}])

st.subheader("Datos enviados al modelo")
st.dataframe(datos)

if st.button("Predecir"):
    prediccion = model.predict(datos)[0]

    if prediccion == 1:
        st.success("Predicción: el estudiante SÍ aprobará.")
    else:
        st.warning("Predicción: el estudiante NO aprobará.")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(datos)[0]
        st.write(f"Probabilidad de NO aprobar: {proba[0]:.4f}")
        st.write(f"Probabilidad de SÍ aprobar: {proba[1]:.4f}")