# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 22:23:59 2025

@author: STUDENT
"""

import numpy as np
import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Loading the trained model
loaded_model = pickle.load(open("C:/Users/STUDENT/Desktop/DRUG PRESCRIPTION USING ML/RandomForest_model.pkl",'rb'))
from sklearn.preprocessing import LabelEncoder
import numpy as np
import streamlit as st

# Initialize encoder
gender_encoder = LabelEncoder()

# Fit the encoder with all possible values
gender_encoder.fit(['F', 'M'])

# Example encoding
print(gender_encoder.transform(['F', 'M']))

def diagnosis(input_data):
    # 1. Access the categorical value (e.g., gender)
    gender = input_data['gender']

    # 2. Encode the categorical value to a number
    encoded_gender = gender_encoder.transform([gender])
    
    # 3. Create the final numpy array for prediction
    # Ensure all other features are also numbers
    processed_input = np.array([[
        input_data['age'],
        input_data['blood_pressure'],
        encoded_gender[0], # Use the encoded value
        # Add other numerical features
    ]])
    
    # 4. Make the prediction with the cleaned data
    # Assuming 'loaded_model' is defined elsewhere
    prediction = loaded_model.predict(processed_input)
    
    return prediction
def main():
    #giving a title
    st.title("Drug Prescription App")
    
    #getting the input data from the user
    Age= st.text_input("Age")
    Sex= st.text_input("Sex")
    BP= st.text_input("Blood Pressure")
    Cholesterol= st.text_input("Cholesterol level")
    Na_to_K= st.text_input("Sodium to Potassium")
    
    input_data= [
        Age,
        Sex,
        BP,
        Cholesterol,
        Na_to_K
        ]
    
    
    if st.button("Drug Prescription Result"):
        diagnosis_result = diagnosis(input_data)
        st.success(diagnosis_result)

if __name__ == '__main__':
    main()
    
    
    