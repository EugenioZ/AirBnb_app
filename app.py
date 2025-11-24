import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================
# CONFIGURACIÓN INICIAL
# ==========================
st.set_page_config(
    page_title="Predicción de precios Airbnb - Buenos Aires",
    page_icon="🏙️",
    layout="centered"
)

st.title("🏙️ Predicción de precios de alojamientos Airbnb (Buenos Aires)")
st.write(
    "Ingresá las características del alojamiento y obtené el **precio estimado por noche** "
    "según un modelo entrenado sobre datos de Airbnb."
)

MODEL_PATH = Path("modelo_lightgbm_mejor.joblib")
ENCODER_PATH = Path("encoder_barrio.joblib")

# ==========================
# CARGA DEL MODELO Y ENCODER
# ==========================
@st.cache_resource
def load_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path.resolve()}")
    return joblib.load(path)

try:
    model = load_file(MODEL_PATH)
    encoder = load_file(ENCODER_PATH)
    loaded = True
except Exception as e:
    st.error("❌ No se pudo cargar el modelo o el encoder.")
    st.code(str(e))
    loaded = False


# IMPORTANT: columnas del modelo (en el orden correcto)
FEATURES = [
    'accommodates',
    'bedrooms',
    'beds',
    'bathrooms',
    'availability_365',
    'review_scores_rating',
    'reviews_per_month',
    'dist_obelisco_km',
    'number_of_reviews',
    'barrio_encoded'             # <--- NUEVA COLUMNA
]

# ==========================
# SIDEBAR: INPUTS DEL USUARIO
# ==========================
st.sidebar.header("✏️ Parámetros del alojamiento")

accommodates = st.sidebar.number_input("Capacidad (personas)", 1, 16, 2)
bedrooms = st.sidebar.number_input("Dormitorios", 0, 10, 1)
beds = st.sidebar.number_input("Camas", 0, 16, 1)
bathrooms = st.sidebar.number_input("Baños", 0.0, 5.0, 1.0, step=0.5)

availability_365 = st.sidebar.number_input("Disponibilidad anual (1–365)", 1, 365, 180)
review_scores_rating = st.sidebar.number_input("Puntaje reviews (1–5)", 1.0, 5.0, 4.5, step=0.1)

reviews_per_month = st.sidebar.number_input("Reviews por mes", 0.0, 15.0, 1.0, step=0.1)
dist_obelisco_km = st.sidebar.number_input("Distancia al Obelisco (km)", 0.0, 30.0, 5.0, step=0.1)
number_of_reviews = st.sidebar.number_input("Número total de reviews", 0, 5000, 10)

# Nuevo input: barrio
barrio = st.sidebar.text_input("Barrio (ej: Palermo, Recoleta, Caballito)", "")


# ==========================
# PROCESAR INPUT + ENCODE
# ==========================

def encode_barrio(input_barrio):
    """
    Toma el barrio que ingresa el usuario y lo convierte en número usando el encoder.
    Si el barrio no existe en el encoder, devuelve un valor neutral (por ejemplo, -1).
    """
    try:
        encoded_value = encoder.transform([input_barrio])[0]
    except Exception:
        encoded_value = -1  # barrio desconocido
    return encoded_value


if barrio.strip() == "":
    st.warning("Ingresá un barrio para poder predecir.")
    barrio_encoded = None
else:
    barrio_encoded = encode_barrio(barrio)


# ==========================
# ARMAR DATAFRAME
# ==========================
if barrio_encoded is not None:
    input_data = pd.DataFrame([{
        'accommodates': accommodates,
        'bedrooms': bedrooms,
        'beds': beds,
        'bathrooms': bathrooms,
        'availability_365': availability_365,
        'review_scores_rating': review_scores_rating,
        'reviews_per_month': reviews_per_month,
        'dist_obelisco_km': dist_obelisco_km,
        'number_of_reviews': number_of_reviews,
        'barrio_encoded': barrio_encoded
    }])[FEATURES]

    with st.expander("🔍 Datos procesados que se envían al modelo"):
        st.dataframe(input_data)


# ==========================
# PREDICCIÓN
# ==========================
st.markdown("---")
st.subheader("📈 Predicción")

if st.button("Predecir precio por noche"):
    if not loaded:
        st.error("No se pudo cargar el modelo o el encoder.")
    elif barrio.strip() == "":
        st.error("Ingresá un barrio válido.")
    else:
        try:
            pred = model.predict(input_data)[0]
            st.success(f"💵 Precio estimado por noche: **ARS {pred:,.2f}**")
        except Exception as e:
            st.error("❌ Error al hacer la predicción.")
            st.code(str(e))
