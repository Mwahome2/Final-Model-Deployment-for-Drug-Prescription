# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 21:39:48 2025

@author: STUDENT
"""

import streamlit as st
import pickle
import pandas as pd
import os
import joblib # Import joblib
import numpy as np # Import numpy

# --- 1. Load Machine Learning Model and Feature Names ---
loaded_model = None
label_encoders = {}  # Dictionary to hold all label encoders

try:
    model_path = '/content/drug_prediction_model.joblib'
    loaded_model = joblib.load(open(model_path, 'rb'))

    # Load all individual encoders
    label_encoders['Sex'] = joblib.load(open('/content/sex_encoder.joblib', 'rb'))
    label_encoders['BP'] = joblib.load(open('/content/bp_encoder.joblib', 'rb'))
    label_encoders['Cholesterol'] = joblib.load(open('/content/cholesterol_encoder.joblib', 'rb'))
    label_encoders['Drug'] = joblib.load(open('/content/drug_label_encoder.joblib', 'rb')) # Load the Drug encoder

    st.success("Machine learning model and label encoders loaded successfully. ✅")
except FileNotFoundError:
    st.error("Error: Ensure model and encoder files are present in the /content/ directory. ❌")
except Exception as e:
    st.error(f"Error loading ML assets: {e} ⚠️")

# --- Prediction Function ---
def predict_drug(input_data_features):
    input_data_as_numpy_array = np.asarray(input_data_features)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    if loaded_model is None:
        return "Model not available"

    try:
        numerical_prediction = loaded_model.predict(input_data_reshaped)
        # Decode the prediction using the 'Drug' encoder
        predicted_drug_name = label_encoders['Drug'].inverse_transform(numerical_prediction)[0]

        return predicted_drug_name
    except Exception as e:
        st.error(f"An error occurred during prediction: {e} 😞")
        return "Prediction error"

# --- Main Streamlit App Function ---
def main():
    st.set_page_config(page_title="Drug Prescription Prediction", layout="centered")

    st.title("💊 AI-Assisted Drug Prescription Prediction Web App")
    st.markdown("---")
    st.write("Enter patient details to get a drug prescription recommendation.")

    st.header("Patient Information")

    age = st.slider("Age", min_value=15, max_value=74, value=30, step=1)

    # Use the loaded encoders to map options to numerical values
    # Ensure the classes_ attribute is available from the loaded encoders
    if 'Sex' in label_encoders and hasattr(label_encoders['Sex'], 'classes_'):
        sex_options = list(label_encoders['Sex'].classes_)
        sex = st.selectbox("Sex", sex_options)
        sex_encoded = label_encoders['Sex'].transform([sex])[0]
    else:
        st.warning("Sex encoder not loaded correctly. Cannot display sex options.")
        sex_encoded = None # Or handle appropriately

    if 'BP' in label_encoders and hasattr(label_encoders['BP'], 'classes_'):
        bp_options = list(label_encoders['BP'].classes_)
        bp = st.selectbox("Blood Pressure", bp_options)
        bp_encoded = label_encoders['BP'].transform([bp])[0]
    else:
         st.warning("BP encoder not loaded correctly. Cannot display BP options.")
         bp_encoded = None # Or handle appropriately

    if 'Cholesterol' in label_encoders and hasattr(label_encoders['Cholesterol'], 'classes_'):
        cholesterol_options = list(label_encoders['Cholesterol'].classes_)
        cholesterol = st.selectbox("Cholesterol level", cholesterol_options)
        cholesterol_encoded = label_encoders['Cholesterol'].transform([cholesterol])[0]
    else:
        st.warning("Cholesterol encoder not loaded correctly. Cannot display Cholesterol options.")
        cholesterol_encoded = None # Or handle appropriately


    na_to_k = st.number_input("Sodium to Potassium", min_value=0.0, max_value=50.0, value=15.0, step=0.1)

    # --- Initialize prediction result variable ---
    predicted_drug_result = ''

    st.markdown("---")

    # Order of input data MUST match the order of features used during model training
    # Assuming the order is 'Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K' based on your notebook.
    input_data_for_prediction = [
        age,
        sex_encoded,
        bp_encoded,
        cholesterol_encoded,
        na_to_k
    ]

    # Only attempt prediction if all encoders loaded correctly and encoded values are not None
    if st.button("Get Drug Recommendation 🩺"):
        if all(v is not None for v in [age, sex_encoded, bp_encoded, cholesterol_encoded, na_to_k]):
            try:
                predicted_drug_result = predict_drug(input_data_for_prediction)

            except Exception as e:
                st.error(f"An error occurred: {e} 😞")
                predicted_drug_result = "Prediction error"
        else:
            st.warning("Please ensure all input fields are valid.")


    if predicted_drug_result and predicted_drug_result not in ["Model not available", "Prediction error"]:
        st.success(f"**Recommended Drug:** <span style='font-size: 2em; font-weight: bold;'>{predicted_drug_result}</span> 🌟", unsafe_allow_html=True)
    elif predicted_drug_result:
        st.error(predicted_drug_result)

    st.markdown("---")
    st.caption("Disclaimer: This prediction is an AI-assisted recommendation and should be thoroughly reviewed by a qualified physician.")

if __name__ == '__main__':
    main()