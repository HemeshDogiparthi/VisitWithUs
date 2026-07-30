
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('VisitWithUs_project/deployment/best_model.pkl')

#Collect the user input into a data frame for the model that is loaded
# Streamlit UI
st.title("Customer Purchase Prediction")
st.write("Enter customer details to predict purchase.")

# User input

ProdTaken = st.selectbox("Product Taken", [0, 1])
Age = st.number_input("Age", min_value=0, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch", min_value=0, max_value=100, value=30)
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer"])
Gender  = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Person Visiting", min_value=0, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number of Followups", min_value=0, max_value=10, value=2)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Premium"])
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips", min_value=0, max_value=10, value=2)
Passport = st.selectbox("Passport", [0, 1])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=0, max_value=10, value=5)
OwnCar = st.selectbox("Own Car", [0, 1])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=2)
MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=100000, value=50000)



# Assemble input into DataFrame
input_data = pd.DataFrame([{
    "ProdTaken": ProdTaken,
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting
    "MonthlyIncome": MonthlyIncome
}])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Predicted Purchase: **{prediction}**")
