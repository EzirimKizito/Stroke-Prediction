import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model and transformers
with open('min_max_scaler.pkl', 'rb') as f:
    min_max_scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('randomforest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set up the Streamlit app
st.title('Stroke Prediction App')
st.markdown("#### A FINAL YEAR PROJECT WORK BY: ELIJAH OYINDAMOLA")
# Create input fields for each feature
gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
age = st.slider('Age', 0.08, 82.0, 42.0)
hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
ever_married = st.selectbox('Ever Married', ['No', 'Yes'])
work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
avg_glucose_level = st.slider('Average Glucose Level', 55.12, 271.74, 105.30)
bmi = st.slider('BMI', 10.3, 97.6, 28.89)
smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

# Create a dataframe from the inputs
input_data = pd.DataFrame({
    'gender': [gender],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'ever_married': [ever_married],
    'work_type': [work_type],
    'Residence_type': [residence_type],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'smoking_status': [smoking_status]
})

# Function to preprocess the input data
def preprocess_input(data):
    # Apply label encoding to categorical variables
    for column in ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
        data[column] = label_encoders[column].transform(data[column])
    
    # Apply Min-Max scaling to numerical variables
    numerical_columns = ['age', 'avg_glucose_level', 'bmi']
    data[numerical_columns] = min_max_scaler.transform(data[numerical_columns])
    
    return data

# Make prediction when the user clicks the button
if st.button('Predict Stroke Risk'):
    processed_input = preprocess_input(input_data)
    prediction = model.predict(processed_input)
    probability = model.predict_proba(processed_input)

    st.subheader('Prediction Result:')
    if prediction[0] == 0:
        st.write('Low risk of stroke')
    else:
        st.write('High risk of stroke')
    
    st.write(f'Probability of stroke: {probability[0][1]:.2%}')

# Add some information about the model and its use
st.sidebar.header('About')
st.sidebar.info('This application uses a Random Forest model to predict the risk of stroke based on various health and lifestyle factors. It is developed as part of a final year project by Elijah Oyindamola, with matric number 19/52HA001. The model considers features such as age, hypertension, heart disease, smoking status, and other relevant factors to provide an assessment of stroke risk.

The goal of this project is to leverage machine learning techniques to aid in early detection and prevention of stroke, potentially contributing to better healthcare outcomes. The application is designed to be user-friendly and accessible, providing valuable insights based on the input data.')
