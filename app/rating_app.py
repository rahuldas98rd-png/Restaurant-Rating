import streamlit as st
import numpy as np
from styles.styles import load_styles
from templates.templates import (
    HERO_HTML, FOOTER_HTML,
    SECTION_LOCATION_HTML, SECTION_COST_HTML,
    SECTION_SERVICE_HTML, SECTION_PRICE_HTML,
    SECTION_CLOSE_HTML, result_banner_html,
)
from src.utils.main_utils.utils import read_yaml_file
from backend import predict_rating

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Restaurant Rating Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_styles()

# ─────────────────────────────────────────────
#  DATA  –  dropdown options
# ─────────────────────────────────────────────
data = read_yaml_file(file_path="app/data.yaml")

COUNTRY_CODES = data['country_code']
CITIES = data['cities']
CURRENCIES = data['currencies']
CUISINES = data['cuisines']
PRICE_RANGE_MAP = data['price_range']


# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown(HERO_HTML, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FORM  –  two-column layout
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # ── Location & Currency ──
    st.markdown(SECTION_LOCATION_HTML, unsafe_allow_html=True)
    country_label = st.selectbox("Country Code", list(COUNTRY_CODES.keys()), index=0)
    country_code  = COUNTRY_CODES[country_label]
    city          = st.selectbox("City", CITIES, index=0)
    currency      = st.selectbox("Currency", CURRENCIES, index=0)
    st.markdown(SECTION_CLOSE_HTML, unsafe_allow_html=True)

    # ── Cost & Votes ──
    st.markdown(SECTION_COST_HTML, unsafe_allow_html=True)
    avg_cost = st.number_input(
        "Average Cost for Two",
        min_value=0.0, max_value=100_000.0,
        value=500.0, step=50.0, format="%.2f",
    )
    votes = st.number_input(
        "Number of Votes",
        min_value=0, max_value=1_000_000,
        value=100, step=10,
    )
    st.markdown(SECTION_CLOSE_HTML, unsafe_allow_html=True)

with col_right:
    # ── Services ──
    st.markdown(SECTION_SERVICE_HTML, unsafe_allow_html=True)
    has_table_booking   = st.radio("Has Table Booking?",   ["Yes", "No"], horizontal=True)
    st.markdown("<br>", unsafe_allow_html=True)
    has_online_delivery = st.radio("Has Online Delivery?", ["Yes", "No"], horizontal=True)
    st.markdown("<br>", unsafe_allow_html=True)
    is_delivering_now   = st.radio("Is Delivering Now?",   ["Yes", "No"], horizontal=True)
    st.markdown(SECTION_CLOSE_HTML, unsafe_allow_html=True)

    # ── Price Range & Cuisine ──
    st.markdown(SECTION_PRICE_HTML, unsafe_allow_html=True)
    price_range_label = st.radio(
        "Price Range",
        list(PRICE_RANGE_MAP.keys()),
        horizontal=True,
    )
    price_range = PRICE_RANGE_MAP[price_range_label]
    cuisines = st.multiselect(
        "Cuisines (select one or more)",
        CUISINES,
        default=["North Indian"],
    )
    if not cuisines:
        st.warning("Please select at least one cuisine.")
    st.markdown(SECTION_CLOSE_HTML, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PREDICT BUTTON  (centred)
# ─────────────────────────────────────────────
_, btn_col, _ = st.columns([1.5, 1, 1.5])
with btn_col:
    predict_clicked = st.button("✦ Predict Rating", use_container_width=True)

# ─────────────────────────────────────────────
#  MOCK MODEL  –  deterministic formula
#  Replace with your real model.predict() call
# ─────────────────────────────────────────────
def predict(
    country_code, city, currency, avg_cost,
    has_table, has_online, delivering_now,
    price_range, cuisine, votes,
):

    rating = predict_rating(country_code=country_code, city=city,
                            cuisines=cuisine,avg_cost_for_two=avg_cost,
                            currency=currency,has_table_booking=has_table,
                            has_online_delivery=has_online,is_delivering_now=delivering_now,
                            price_range=price_range, votes=votes)

    return rating


def rating_verdict(r):
    if r >= 4.5: return "⭐ Exceptional — Top-tier dining experience"
    if r >= 4.0: return "Very Good — Highly recommended"
    if r >= 3.5: return "Good — Worth a visit"
    if r >= 3.0: return "Average — Decent, room to improve"
    return "Below Average — Needs significant improvement"


def stars(r):
    full  = int(r)
    half  = 1 if (r - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty


# ─────────────────────────────────────────────
#  RESULT
# ─────────────────────────────────────────────
if predict_clicked:
    if not cuisines:
        st.error("Please select at least one cuisine before predicting.")
    else:
        cuisine_str = ", ".join(cuisines)
        rating = predict(
            country_code, city, currency, avg_cost,
            has_table_booking, has_online_delivery, is_delivering_now,
            price_range, cuisine_str, votes,
        )
        verdict  = rating_verdict(rating)
        star_str = stars(rating)
        whole    = int(rating)
        decimal  = str(rating).split(".")[-1]

        st.markdown(
            result_banner_html(
                whole, decimal, star_str, verdict,
                city, cuisine_str, price_range_label.split()[0], votes,
            ),
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown(FOOTER_HTML, unsafe_allow_html=True)
