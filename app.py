import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('restaurant_rating_model.pkl', 'rb'))
        return model
    except:
        return None

model = load_model()

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Restaurant Rating Predictor", layout="wide")

st.title("🍽️ Restaurant Rating Predictor")
st.markdown("Predict restaurant ratings using Machine Learning")

st.sidebar.header("Input Features")

# User Inputs
price_range = st.sidebar.slider("Price Range (1 = Low, 4 = High)", 1, 4, 2)
votes = st.sidebar.number_input("Number of Votes", min_value=0, max_value=10000, value=100)
location = st.sidebar.selectbox("Location Tier", ["Tier 1", "Tier 2", "Tier 3"])
online_order = st.sidebar.selectbox("Online Order Available", ["Yes", "No"])
book_table = st.sidebar.selectbox("Table Booking Available", ["Yes", "No"])

# Feature Encoding
location_map = {"Tier 1": 2, "Tier 2": 1, "Tier 3": 0}
online_order_map = {"Yes": 1, "No": 0}
book_table_map = {"Yes": 1, "No": 0}

input_data = pd.DataFrame({
    'price_range': [price_range],
    'votes': [votes],
    'location': [location_map[location]],
    'online_order': [online_order_map[online_order]],
    'book_table': [book_table_map[book_table]]
})

# -----------------------------
# Prediction
# -----------------------------
st.subheader("Prediction")

if st.button("Predict Rating"):
    if model is not None:
        prediction = model.predict(input_data)[0]
        st.success(f"⭐ Predicted Rating: {round(prediction, 2)}")
    else:
        st.error("Model not found. Please upload 'restaurant_rating_model.pkl'")

# -----------------------------
# Data Preview Section
# -----------------------------
st.subheader("Sample Input Data")
st.write(input_data)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Portfolio Project")

