import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.title("Holiday Package Prediction")

age = st.slider("Age", 18, 100, value=30)
typeofcontact = st.radio("Type of Contact", options=["Self Enquiry", "Company Invited"])
citytier = st.selectbox("City Tier", options=[1, 2, 3])
durationofpitch = st.slider("Duration of Pitch (minutes)", 0, 60, value=30)
occupation = st.selectbox(
    "Occupation",
    options=["Salaried", "Free Lancer", "Small Business", "Large Business"],
)
gender = st.radio("Gender", options=["Male", "Female"])
numberoffollowups = st.number_input(
    "Number of Follow-ups", min_value=0.0, step=1.0, value=2.0
)
productpitched = st.selectbox(
    "Product Pitched", options=["Deluxe", "Basic", "Standard", "Super Deluxe", "King"]
)
preferredpropertystar = st.select_slider(
    "Preferred Property Star", options=[3.0, 4.0, 5.0], value=4.0
)
maritalstatus = st.radio("Marital Status", options=["Unmarried", "Married", "Divorced"])
numberoftrips = st.number_input("Number of Trips", min_value=0.0, step=1.0, value=1.0)
passport = st.radio("Passport", options=["Yes", "No"])
pitchsatisfactionscores = st.select_slider(
    "Pitch Satisfaction Score", options=[1, 2, 3, 4, 5], value=3
)
owncar = st.radio("Own Car", options=["Yes", "No"])
designation = st.selectbox(
    "Designation", options=["Manager", "Executive", "Senior Manager", "AVP", "VP"]
)
monthlyincome = st.number_input(
    "Monthly Income (in currency)", min_value=0.0, step=1000.0, value=50000.0
)
totalvisiting = st.number_input(
    "Total Number of People Visiting", min_value=0, step=1, value=1
)

if st.button("Predict"):
    try:
        input_data = {
            "age": age,
            "typeofcontact": typeofcontact,
            "citytier": citytier,
            "durationofpitch": durationofpitch,
            "occupation": occupation,
            "gender": gender,
            "numberoffollowups": numberoffollowups,
            "productpitched": productpitched,
            "preferredpropertystar": preferredpropertystar,
            "maritalstatus": maritalstatus,
            "numberoftrips": numberoftrips,
            "passport": 1 if passport == "Yes" else 0,
            "pitchsatisfactionscores": pitchsatisfactionscores,
            "owncar": 1 if owncar == "Yes" else 0,
            "designation": designation,
            "monthlyincome": monthlyincome,
            "totalvisiting": totalvisiting,
        }

        response = requests.post(API_URL, json=input_data)
        response_data = response.json()

        if response.status_code == 200:
            prediction = response_data.get("prediction", None)
            if prediction == 1:
                st.success(response_data["prediction_message"])
            elif prediction == 0:
                st.error(response_data["prediction_message"])
            else:
                st.warning("Uh oh! Something went wrong.")
        else:
            st.error(f"Error from API: {response_data.get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
