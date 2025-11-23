import streamlit as st
import pandas as pd
import joblib

# ====== Cargar el modelo ======
@st.cache_resource
def load_model():
    return joblib.load("modelo_lightgbm_mejor.joblib")

model = load_model()

FEATURES = [
    'accommodates',
    'bedrooms',
    'beds',
    'bathrooms',
    'availability_365',
    'review_scores_rating',
    'reviews_per_month',
    'dist_obelisco_km',
    'number_of_reviews'
]

# ====== Interfaz ======
st.title("Predicción de precios de Airbnb (Buenos Aires)")
st.write("Ingresá las características y obtené el precio estimado por noche.")

# Campos del usuario
accommodates = st.number_input("Capacidad (accommodates)", min_value=1, max_value=20, value=2)
bedrooms = st.number_input("Dormitorios (bedrooms)", min_value=0, max_value=10, value=1)
beds = st.number_input("Camas (beds)", min_value=0, max_value=20, value=1)
bathrooms = st.number_input("Baños (bathrooms)", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
availability_365 = st.number_input("Disponibilidad anual (availability_365)", min_value=0, max_value=365, value=180)
review_scores_rating = st.number_input("Puntaje de reviews (review_scores_rating)", min_value=0.0, max_value=100.0, value=90.0)
reviews_per_month = st.number_input("Reviews por mes (reviews_per_month)", min_value=0.0, max_value=50.0, value=1.0)
dist_obelisco_km = st.number_input("Distancia al Obelisco (km)", min_value=0.0, max_value=50.0, value=5.0)
number_of_reviews = st.number_input("Número total de reviews", min_value=0, max_value=5000, value=10)

if st.button("Predecir precio"):
    data = pd.DataFrame([{
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "beds": beds,
        "bathrooms": bathrooms,
        "availability_365": availability_365,
        "review_scores_rating": review_scores_rating,
        "reviews_per_month": reviews_per_month,
        "dist_obelisco_km": dist_obelisco_km,
        "number_of_reviews": number_of_reviews
    }])

    data = data[FEATURES]
    prediction = model.predict(data)[0]

    st.subheader("Precio estimado por noche:")
    st.success(f"ARS {prediction:,.2f}")

