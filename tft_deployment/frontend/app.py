import streamlit as st
import requests
import json
import numpy as np

st.title("📊 Walmart Sales Forecast - TFT Inspired LSTM")

st.markdown("Enter values for the past 24 weeks (6 features each):")
st.caption("Temperature, Fuel_Price, CPI, Unemployment, IsHoliday (0/1), Dummy Revenue")
sequence = []

num_weeks = 24
num_features = 6

for i in range(num_weeks):
    row = []
    st.subheader(f"Week {i+1}")
    for j, feature_name in enumerate(["Temperature", "Fuel_Price", "CPI", "Unemployment", "IsHoliday", "DummyRevenue"]):
        val = st.number_input(f"{feature_name} (Week {i+1})", key=f"{i}_{j}")
        row.append(val)
    sequence.append(row)

if st.button("🚀 Predict Next 12 Weeks"):
    with st.spinner("Sending data to model..."):
        res = requests.post("http://127.0.0.1:8000/predict", json={"sequence": sequence})
        if res.status_code == 200:
            prediction = res.json()["prediction"]
            st.success("✅ Forecast ready!")
            st.line_chart(prediction)
        else:
            st.error("❌ Prediction failed!")