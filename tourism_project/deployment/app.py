import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_wellness_tourism_model_v1.joblib",
)

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Wellness Tourism Prediction",
    page_icon="✈️",
)

st.title("Wellness Tourism Package Purchase Prediction")
st.write(
    "Enter customer and interaction details to estimate whether the "
    "customer is likely to purchase the Wellness Tourism Package."
)

Age = st.number_input("Age", 18, 100, 35)
TypeofContact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Enquiry"],
)
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business"],
)
Gender = st.selectbox("Gender", ["Male", "Female", "Fe Male"])
NumberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting", 1, 10, 2
)
PreferredPropertyStar = st.selectbox(
    "Preferred Property Star", [3, 4, 5]
)
MaritalStatus = st.selectbox(
    "Marital Status",
    ["Married", "Single", "Divorced", "Unmarried"],
)
NumberOfTrips = st.number_input(
    "Number of Trips per Year", 0.0, 20.0, 3.0, 0.5
)
Passport = st.selectbox("Passport", [0, 1])
OwnCar = st.selectbox("Own Car", [0, 1])
NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting", 0, 10, 0
)
Designation = st.selectbox(
    "Designation",
    ["AVP", "Manager", "Executive", "Senior Manager", "VP"],
)
MonthlyIncome = st.number_input(
    "Monthly Income", 0.0, 1000000.0, 25000.0, 500.0
)
PitchSatisfactionScore = st.selectbox(
    "Pitch Satisfaction Score", [1, 2, 3, 4, 5]
)
ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"],
)
NumberOfFollowups = st.number_input("Number of Followups", 0, 10, 3)
DurationOfPitch = st.number_input(
    "Duration of Pitch (minutes)", 0.0, 120.0, 15.0, 1.0
)

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "ProductPitched": ProductPitched,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch,
}])

if st.button("Predict Purchase"):
    probability = model.predict_proba(input_data)[0, 1]
    prediction = int(probability >= 0.45)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            "The model predicts that the customer is **likely to "
            "purchase** the Wellness Tourism Package."
        )
    else:
        st.info(
            "The model predicts that the customer is **unlikely to "
            "purchase** the Wellness Tourism Package."
        )

    st.write(f"Estimated purchase probability: **{probability:.2%}**")
