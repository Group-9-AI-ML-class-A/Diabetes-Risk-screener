import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Diabetes Risk Screener")

@st.cache_resource
def load_all():
    scaler = joblib.load("diabetes_scaler.pkl")
    balanced = joblib.load("diabetes_risk_model_balanced.pkl")
    high_recall = joblib.load("diabetes_risk_model_high_recall.pkl")
    return scaler, balanced, high_recall

scaler, model_balanced, model_high_recall = load_all()

COLUMNS = [
    'age', 'bmi', 'hours_sleep_per_night', 'stress_level',
    'blood_pressure_systolic', 'blood_pressure_diastolic',
    'waist_circumference_cm', 'gender_Male', 'gender_Other',
    'family_history_diabetes_Yes', 'physical_activity_level_Moderate',
    'physical_activity_level_Sedentary', 'diet_type_Pescatarian',
    'diet_type_Vegan', 'diet_type_Vegetarian', 'smoking_status_Former',
    'smoking_status_Never', 'alcohol_consumption_Occasional',
    'alcohol_consumption_Regular', 'alcohol_consumption_Unknown'
]

HIGH_RECALL_CUTOFF = 0.30

st.title("Diabetes Risk Screener")
st.caption("A screening aid only. It does not diagnose diabetes.")

mode = st.radio(
    "Select model:",
    ("Balanced", "High recall (catches more cases, more false alarms)")
)
st.markdown("---")

st.subheader("Patient details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=40)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    waist = st.number_input("Waist circumference (cm)", min_value=40.0, max_value=200.0, value=85.0)
    sleep = st.number_input("Hours of sleep per night", min_value=0.0, max_value=16.0, value=7.0)
    stress = st.slider("Stress level", min_value=1, max_value=10, value=5)
    systolic = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=250, value=120)
    diastolic = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)

with col2:
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    family_history = st.selectbox("Family history of diabetes", ["No", "Yes"])
    activity = st.selectbox("Physical activity level", ["Active", "Moderate", "Sedentary"])
    diet = st.selectbox("Diet type", ["Omnivore", "Pescatarian", "Vegan", "Vegetarian"])
    smoking = st.selectbox("Smoking status", ["Current", "Former", "Never"])
    alcohol = st.selectbox("Alcohol consumption", ["None", "Occasional", "Regular", "Unknown"])

if st.button("Calculate risk"):
    row = {
        'age': age,
        'bmi': bmi,
        'hours_sleep_per_night': sleep,
        'stress_level': stress,
        'blood_pressure_systolic': systolic,
        'blood_pressure_diastolic': diastolic,
        'waist_circumference_cm': waist,
        'gender_Male': int(gender == "Male"),
        'gender_Other': int(gender == "Other"),
        'family_history_diabetes_Yes': int(family_history == "Yes"),
        'physical_activity_level_Moderate': int(activity == "Moderate"),
        'physical_activity_level_Sedentary': int(activity == "Sedentary"),
        'diet_type_Pescatarian': int(diet == "Pescatarian"),
        'diet_type_Vegan': int(diet == "Vegan"),
        'diet_type_Vegetarian': int(diet == "Vegetarian"),
        'smoking_status_Former': int(smoking == "Former"),
        'smoking_status_Never': int(smoking == "Never"),
        'alcohol_consumption_Occasional': int(alcohol == "Occasional"),
        'alcohol_consumption_Regular': int(alcohol == "Regular"),
        'alcohol_consumption_Unknown': int(alcohol == "Unknown"),
    }

    patient = pd.DataFrame([row])[COLUMNS]

    try:
        scaled = scaler.transform(patient)

        if mode == "Balanced":
            label = str(model_balanced.predict(scaled)[0])
        else:
            idx = list(model_high_recall.classes_).index("At_Risk")
            p_at_risk = model_high_recall.predict_proba(scaled)[0][idx]
            label = "At_Risk" if p_at_risk >= HIGH_RECALL_CUTOFF else "Low"

        st.subheader("Result")
        if label == "Low":
            st.success("🟢 Low risk")
        elif label == "Moderate":
            st.warning("🟡 Moderate risk")
        elif label == "High":
            st.error("🔴 High risk")
        else:
            st.error("🔴 At risk. Consider confirmatory lab testing.")
    except ValueError as e:
        st.error(f"Column mismatch: {e}")