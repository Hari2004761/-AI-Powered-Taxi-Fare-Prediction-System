import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('taxifare_model.pkl')

feature_cols = ['passenger_count', 'trip_distance', 'duration', 'payment_type_Card', 'payment_type_Cash']

st.title("Taxi Fare Prediction 🚖")
st.write("Enter the details of your ride below:")

# Input fields
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=10, value=1)
trip_distance = st.number_input("Trip Distance (km)", min_value=0.1, value=5.0)
duration = st.number_input("Trip Duration (minutes)", min_value=1, value=15)
payment_type = st.selectbox("Payment Type", ['Cash', 'Card'])

# Create input DataFrame with all zeros for features
input_data = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)

input_data['passenger_count'] = passenger_count
input_data['trip_distance'] = trip_distance
input_data['duration'] = duration

if payment_type == 'Cash':
    input_data['payment_type_Cash'] = 1
elif payment_type == 'Card':
    input_data['payment_type_Card'] = 1

if st.button("Predict Fare"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Fare: ${prediction:.2f}")
