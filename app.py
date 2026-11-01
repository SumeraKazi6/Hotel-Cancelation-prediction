import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Hotel Cancellation Predictor", page_icon="🏨", layout="wide")

import urllib.request
import shutil

MODEL_PATH = Path(__file__).parent / "hotel_cancellation_model.pkl"

MODEL_URL = "https://github.com/SumeraKazi6/Hotel-Cancelation-prediction/releases/download/v1.0/hotel_cancellation_model.pkl"

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        with urllib.request.urlopen(MODEL_URL, timeout=600) as response:
            with open(MODEL_PATH, "wb") as file:
                shutil.copyfileobj(response, file)

    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🏨 Hotel Booking Cancellation Predictor")
st.write("Enter reservation details to estimate whether the booking is likely to be cancelled.")

if model is None:
    st.error("Model file not found. Add hotel_cancellation_model.pkl to this project folder. See README.md for instructions.")
    st.stop()

with st.form("booking_form"):
    st.subheader("Hotel and arrival details")
    c1, c2, c3 = st.columns(3)
    with c1:
        hotel = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
        lead_time = st.number_input("Lead time (days)", min_value=0, value=30)
        arrival_date_year = st.selectbox("Arrival year", [2015, 2016, 2017])
        arrival_date_month = st.selectbox("Arrival month", ["January","February","March","April","May","June","July","August","September","October","November","December"])
        arrival_date_week_number = st.number_input("Arrival week number", min_value=1, max_value=53, value=20)
        arrival_date_day_of_month = st.number_input("Arrival day of month", min_value=1, max_value=31, value=15)
    with c2:
        stays_in_weekend_nights = st.number_input("Weekend nights", min_value=0, value=1)
        stays_in_week_nights = st.number_input("Week nights", min_value=0, value=2)
        adults = st.number_input("Adults", min_value=0, value=2)
        children = st.number_input("Children", min_value=0, value=0)
        babies = st.number_input("Babies", min_value=0, value=0)
        meal = st.selectbox("Meal", ["BB", "HB", "SC", "FB", "Undefined"])
    with c3:
        country = st.text_input("Country code", value="PRT", max_chars=3).upper()
        market_segment = st.selectbox("Market segment", ["Online TA","Offline TA/TO","Groups","Direct","Corporate","Complementary","Aviation","Undefined"])
        distribution_channel = st.selectbox("Distribution channel", ["TA/TO","Direct","Corporate","GDS","Undefined"])
        is_repeated_guest = st.selectbox("Repeated guest?", [0,1])
        reserved_room_type = st.text_input("Reserved room type", value="A", max_chars=2).upper()
        assigned_room_type = st.text_input("Assigned room type", value="A", max_chars=2).upper()

    st.subheader("Booking and customer details")
    d1, d2, d3 = st.columns(3)
    with d1:
        deposit_type = st.selectbox("Deposit type", ["No Deposit","Non Refund","Refundable"])
        customer_type = st.selectbox("Customer type", ["Transient","Transient-Party","Contract","Group"])
        previous_cancellations = st.number_input("Previous cancellations", min_value=0, value=0)
        previous_bookings_not_canceled = st.number_input("Previous bookings not cancelled", min_value=0, value=0)
    with d2:
        booking_changes = st.number_input("Booking changes", min_value=0, value=0)
        agent = st.number_input("Agent ID (0 if none)", min_value=0, value=0)
        days_in_waiting_list = st.number_input("Days in waiting list", min_value=0, value=0)
        adr = st.number_input("Average daily rate (ADR)", min_value=0.0, value=100.0, step=5.0)
    with d3:
        required_car_parking_spaces = st.number_input("Required parking spaces", min_value=0, value=0)
        total_of_special_requests = st.number_input("Special requests", min_value=0, value=0)

    submitted = st.form_submit_button("Predict cancellation", type="primary")

if submitted:
    total_stay = int(stays_in_weekend_nights + stays_in_week_nights)
    total_guests = int(adults + children + babies)
    total_previous_bookings = int(previous_cancellations + previous_bookings_not_canceled)

    # The feature names and derived variables match the project's training notebook.
    booking = pd.DataFrame([{
        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_year": arrival_date_year,
        "arrival_date_month": arrival_date_month,
        "arrival_date_week_number": arrival_date_week_number,
        "arrival_date_day_of_month": arrival_date_day_of_month,
        "stays_in_weekend_nights": stays_in_weekend_nights,
        "stays_in_week_nights": stays_in_week_nights,
        "adults": adults,
        "children": children,
        "babies": babies,
        "meal": meal,
        "country": country if country else "PRT",
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": is_repeated_guest,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "reserved_room_type": reserved_room_type,
        "assigned_room_type": assigned_room_type,
        "deposit_type": deposit_type,
        "customer_type": customer_type,
        "booking_changes": booking_changes,
        "agent": agent,
        "days_in_waiting_list": days_in_waiting_list,
        "adr": adr,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests,
        "total_stay": total_stay,
        "total_guests": total_guests,
        "total_previous_bookings": total_previous_bookings
    }])

    try:
        prediction = int(model.predict(booking)[0])
        st.divider()
        if prediction == 1:
            st.error("Prediction: Booking is likely to be CANCELLED")
        else:
            st.success("Prediction: Booking is likely to be NOT CANCELLED")

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(booking)[0][1])
            st.metric("Estimated cancellation probability", f"{probability:.1%}")
            st.progress(min(max(probability, 0.0), 1.0))
        else:
            st.info("This trained estimator does not expose calibrated prediction probabilities.")
        st.caption("Academic demonstration only. Predictions are estimates, not guarantees or a basis for automatically denying service.")
    except Exception as exc:
        st.exception(exc)
