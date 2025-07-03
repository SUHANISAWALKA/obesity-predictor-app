import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load model and scalers
model = tf.keras.models.load_model("obesity_ann_model.keras")
scaler = joblib.load("obesity_scaler.pkl")
ohe = joblib.load("obesity_ohe.pkl")

# App layout
st.set_page_config(page_title="Obesity Predictor", layout="centered")
st.markdown("""
    <style>
    .main { background-color: #f6f6f9; }
    .stButton>button { background-color: #6a5acd; color: white; font-weight: bold; }
    .st-bb { background-color: white; border-radius: 12px; padding: 20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
    .result-box { background-color: #e6e6fa; padding: 15px; border-radius: 10px; font-size: 18px; font-weight: bold; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("💪 Obesity Risk Predictor")
st.write("Input your health and lifestyle details below to predict your obesity category.")

# Input fields
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        Age = st.slider("Age", 10, 100, 25)
        Height = st.number_input("Height (in meters)", 1.0, 2.5, 1.65, step=0.01)
        Weight = st.number_input("Weight (in kg)", 30.0, 200.0, 60.0, step=0.5)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        family_history = st.selectbox("Family History with Overweight", ["yes", "no"])
        FAVC = st.selectbox("Frequent High Calorie Food", ["yes", "no"])
    
    with col2:
        FCVC = st.slider("Vegetable Consumption (0-3)", 0.0, 3.0, 2.0, step=0.1)
        NCP = st.slider("Main Meals Per Day", 1.0, 4.0, 3.0, step=0.5)
        CAEC = st.selectbox("Eating Between Meals", ["no", "Sometimes", "Frequently", "Always"])
        SMOKE = st.selectbox("Do You Smoke?", ["yes", "no"])
        CH2O = st.slider("Daily Water Intake (Litres)", 0.0, 3.0, 2.0, step=0.1)
        MTRANS = st.selectbox("Transport Mode", ["Automobile", "Bike", "Walking", "Public_Transportation", "Motorbike"])

    submit = st.form_submit_button("Predict Obesity Category")

if submit:
    # Format user input
    input_data = np.array([[Gender, Age, Height, Weight, family_history, FAVC, FCVC, NCP,
                            CAEC, SMOKE, CH2O, MTRANS]])

    # Prepare input as DataFrame to mimic original processing
    import pandas as pd
    input_df = pd.DataFrame(input_data, columns=['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
                                                 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'MTRANS'])

    # One-hot encode categorical features
    cat_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'MTRANS']
    input_df_encoded = pd.get_dummies(input_df, columns=cat_features)

    # Align input features with training set (missing columns = 0)
    expected_columns = scaler.feature_names_in_
    for col in expected_columns:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0
    input_df_encoded = input_df_encoded[expected_columns]

    # Scale input
    input_scaled = scaler.transform(input_df_encoded)

    # Predict
    pred = model.predict(input_scaled)
    pred_label = ohe.inverse_transform(pred)[0][0]
    confidence = float(np.max(pred)) * 100

    # Add description for predicted class
    desc = {
        "Insufficient_Weight": "You are underweight. Consider a balanced diet plan and medical guidance.",
        "Normal_Weight": "You have a healthy weight. Maintain your lifestyle!",
        "Overweight_Level_I": "Mild overweight. Regular activity and mindful eating may help.",
        "Overweight_Level_II": "Moderate overweight. Lifestyle changes are advisable.",
        "Obesity_Type_I": "Obese Class I. Please consider consulting a healthcare professional.",
        "Obesity_Type_II": "Obese Class II. High health risks. Medical guidance recommended.",
        "Obesity_Type_III": "Obese Class III (severe). Immediate professional attention advised."
    }

    st.markdown("---")
    st.markdown(f"""
        <div class="result-box">
            🩺 <strong>Predicted Category:</strong> {pred_label} <br>
            🎯 <strong>Confidence:</strong> {confidence:.2f}% <br><br>
            {desc[pred_label]}
        </div>
    """, unsafe_allow_html=True)
