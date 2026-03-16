import streamlit as st
import joblib
import numpy as np

model = joblib.load("model/business_success_model.pkl")

st.title("Local Business Success Prediction Using AI")

business_type = st.selectbox("Business Type", ["Grocery","Salon","TeaShop","MobileShop","Tailoring"])
location_type = st.selectbox("Location Type", ["Market","Residential","Roadside"])

rent = st.number_input("Monthly Rent (₹)", 1000, 50000)
competition = st.number_input("Nearby Competitors", 0, 20)
footfall = st.number_input("Daily Footfall", 10, 1000)
investment = st.number_input("Initial Investment (₹)", 10000, 1000000)
population = st.number_input("Area Population", 1000, 100000)
online = st.selectbox("Online Presence", ["No","Yes"])

business_map = {"Grocery":0,"Salon":1,"TeaShop":2,"MobileShop":3,"Tailoring":4}
location_map = {"Market":0,"Residential":1,"Roadside":2}
online = 1 if online == "Yes" else 0

input_data = np.array([[
    business_map[business_type],
    location_map[location_type],
    rent, competition, footfall, investment, population, online
]])

if st.button("Predict Success"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("High Chance of Business Success")
    else:
        st.error("High Risk of Business Failure")
