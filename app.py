# Gender -> 1 Female 0 Male
# Churn -> 1 Yes 0 No
# Scaler is exported as scaler
# Model is exported as model.pkl
# Order of the x -> 'Age', 'Gender', 'Tenure', 'MonthlyCharges'



import streamlit as st
import joblib
import numpy as np

# Load scaler and model
scaler = joblib.load("scaler.pkl")   # make sure this file exists
model = joblib.load("model.pkl")     # make sure this file exists

st.title("Churn Prediction App")

st.divider()
st.write("Please enter the values and hit the predict button for getting a prediction.")

st.divider()
age = st.number_input("Enter Age", min_value=10, max_value=100, value=30)
tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)
monthlycharge = st.number_input("Enter Monthly Charge", min_value=30, max_value=150)
gender = st.selectbox("Enter the Gender", ["Male", "Female"])

st.divider()
predictbutton = st.button("Predict")
st.divider()
if predictbutton:
    gender_selected = 1 if gender == "Female" else 0
    x = [age, gender_selected, tenure, monthlycharge]
    x1 = np.array(x)
    x_array = scaler.transform([x1])   # Scale input
    prediction = model.predict(x_array)[0]   # Predict
    predicted = "Yes" if prediction == 1 else "No"
    st.success(f"Churn Prediction: {predicted}")
    st.balloons()

else:
    st.write("Please enter the values and use the Predict button")
