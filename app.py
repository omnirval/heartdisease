# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import joblib

# Set page config and title early
st.set_page_config(page_title="Heart Disease Risk Predictor By Samadhan", page_icon="❤️", layout="centered")

# Custom CSS for clean UI styling
st.markdown("""
<style>
    h1 {
        color: #e63946;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #e63946;
        color: white;
        font-weight: bold;
        width: 100%;
        border-radius: 8px;
        border: none;
        padding: 10px;
        font-size: 1.1rem;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #d62828;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(230, 57, 70, 0.2);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Heart Disease Risk Predictor By Samadhan</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Provide patient clinical and physical details to predict heart disease probability</p>", unsafe_allow_html=True)

# Load saved model, scaler, and expected columns
try:
    model = joblib.load("heart_model.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
except FileNotFoundError:
    st.error("⚠️ Model file(s) not found. Please run the training script in hearttrain.ipynb first to generate heart_model.pkl, scaler.pkl, and columns.pkl.")
    st.stop()

# Organize inputs into columns to prevent long vertical scrolling
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 Demographics & Vitals")
    age = st.slider("Age (years)", 18, 100, 45)
    sex = st.selectbox("Sex", ["M", "F"], help="Patient's physiological sex")
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120, help="Blood pressure at rest")
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 220, help="Total serum cholesterol")
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
    fasting_bs_val = 1 if fasting_bs == "Yes" else 0

with col2:
    st.markdown("### 🩺 Clinical Measurements")
    chest_pain = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"], help="ASY: Asymptomatic, ATA: Atypical Angina, NAP: Non-Anginal Pain, TA: Typical Angina")
    max_hr = st.slider("Max Heart Rate (bpm)", 60, 220, 140, help="Maximum heart rate achieved during exercise")
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["N", "Y"], help="Whether physical exertion triggers chest pain")
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, help="ST depression induced by exercise relative to rest")
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], help="Slope of the peak exercise ST segment")
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], help="Resting electrocardiogram results")

# Prediction action
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Predict"):
    # Create the encoded feature dictionary explicitly to prevent mismatches
    features = {col: 0 for col in expected_columns}

    # Set numeric columns directly
    features['Age'] = age
    features['RestingBP'] = resting_bp
    features['Cholesterol'] = cholesterol
    features['FastingBS'] = fasting_bs_val
    features['MaxHR'] = max_hr
    features['Oldpeak'] = oldpeak

    # Map dummy categories explicitly
    if sex == 'M':
        features['Sex_M'] = 1

    if chest_pain == 'ATA':
        features['ChestPainType_ATA'] = 1
    elif chest_pain == 'NAP':
        features['ChestPainType_NAP'] = 1
    elif chest_pain == 'TA':
        features['ChestPainType_TA'] = 1

    if resting_ecg == 'Normal':
        features['RestingECG_Normal'] = 1
    elif resting_ecg == 'ST':
        features['RestingECG_ST'] = 1

    if exercise_angina == 'Y':
        features['ExerciseAngina_Y'] = 1

    if st_slope == 'Flat':
        features['ST_Slope_Flat'] = 1
    elif st_slope == 'Up':
        features['ST_Slope_Up'] = 1

    # Create input DataFrame
    input_df = pd.DataFrame([features])

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction using probability score
    probability = model.predict_proba(scaled_input)[0][1]
    risk_percentage = probability * 100

    # Display prediction results
    st.markdown("---")
    st.subheader("Prediction Result")

    # Color-coded risk levels
    if risk_percentage < 30.0:
        st.success(f"💚 **Low Risk of Heart Disease** ({risk_percentage:.1f}%)")
        st.markdown("The patient's measurements indicate a low probability of heart disease. Continue maintaining a healthy lifestyle.")
    elif risk_percentage < 70.0:
        st.warning(f"⚠️ **Moderate Risk of Heart Disease** ({risk_percentage:.1f}%)")
        st.markdown("There are moderate risk factors present. It is recommended to monitor vitals closely and consult a physician.")
    else:
        st.error(f"🚨 **High Risk of Heart Disease** ({risk_percentage:.1f}%)")
        st.markdown("The patient's clinical signs indicate a high risk of heart disease. Please consult with a healthcare professional/cardiologist.")