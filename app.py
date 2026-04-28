import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# --- PAGE SETTINGS ---
# This must be the first Streamlit command!
st.set_page_config(page_title="AI Heart Doc", page_icon="🫀", layout="wide")

# --- CUSTOM CSS (For a modern button) ---
st.markdown("""
    <style>
        .stButton>button { width: 100%; border-radius: 8px; border: 2px solid #ff4b4b; color: white; background-color: #ff4b4b; transition: 0.3s; font-size: 18px; font-weight: bold;}
        .stButton>button:hover { background-color: white; color: #ff4b4b; }
    </style>
""", unsafe_allow_html=True)

# 1. Load your trained AI and Data Factory
# (@st.cache_resource tells Streamlit to load this once and keep it in memory, making the app much faster!)
@st.cache_resource
def load_system():
    model = joblib.load("heart_disease_model.pkl")
    pipeline = joblib.load("heart_data_pipeline.pkl")
    return model, pipeline

model, pipeline = load_system()

# 2. Build the Header
st.title("🫀 AI Cardiology Assistant")
st.markdown("**An advanced Machine Learning diagnostic tool.** Enter patient clinical metrics below.")
#st.info("ℹ️ **Medical Disclaimer:** This tool is for educational and portfolio purposes. It should not replace professional medical advice.")

st.divider() # Draws a clean horizontal line

# 3. Organize Inputs into 3 Clean Columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographics & Symptoms")
    age = st.slider("Age", 20, 100, 50, help="Patient's age in years")
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"], help="ATA: Atypical, NAP: Non-Anginal, ASY: Asymptomatic, TA: Typical Angina")
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])

with col2:
    st.subheader("🩸 Vitals & Lab Results")
    # Changed to number_input so users can type numbers directly instead of dragging sliders
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol (mm/dl)", min_value=0, max_value=600, value=200, help="Leave at 0 if unknown")
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    max_hr = st.slider("Maximum Heart Rate Achieved", 60, 200, 140)

with col3:
    st.subheader("📈 ECG Metrics")
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.divider()

# 4. Center the Diagnose Button
_, center_col, _ = st.columns([1, 2, 1]) # Gives the middle column more space

with center_col:
    if st.button("🩺 Run AI Diagnosis"):
        
        # Creates a cool spinning animation while the AI "thinks"
        with st.spinner("Analyzing patient data against clinical patterns..."):
            time.sleep(1) # Fake 1-second delay so the user sees the spinner (UX trick!)
            
            # Bundle data
            user_data = pd.DataFrame({
                "Age": [age], "Sex": [sex], "ChestPainType": [chest_pain],
                "RestingBP": [resting_bp], "Cholesterol": [cholesterol],
                "FastingBS": [fasting_bs], "RestingECG": [resting_ecg],
                "MaxHR": [max_hr], "ExerciseAngina": [exercise_angina],
                "Oldpeak": [oldpeak], "ST_Slope": [st_slope]
            })
            
            # Process data
            user_data[["Cholesterol", "RestingBP"]] = user_data[["Cholesterol", "RestingBP"]].replace(0, np.nan)
            prepared_data = pipeline.transform(user_data)
            
            # Predict
            probabilities = model.predict_proba(prepared_data)
            sick_probability = probabilities[0][1] 
            
        # 5. Display the result beautifully
        st.markdown("### 📊 Diagnostic Result")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            # Streamlit's built-in beautiful dashboard metric
            st.metric(
                label="AI Risk Assessment", 
                value=f"{sick_probability * 100:.1f}%", 
                delta="High Risk" if sick_probability >= 0.30 else "Low Risk", 
                delta_color="inverse"
            )
        
        with res_col2:
            if sick_probability >= 0.30:
                st.error("🚨 **HIGH RISK DETECTED** \n\nThe AI detects patterns strongly consistent with Heart Disease. Immediate consultation with a cardiologist is recommended.")
            else:
                st.success("✅ **LOW RISK DETECTED** \n\nThe AI predicts a healthy heart profile based on the provided metrics.")