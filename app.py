import tensorflow as tf 
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('churn_model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)   

with open ('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


st.title("Customer Churn Prediction")
# User Inputs
geography = st.selectbox("Select Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Select Gender", ["Male", "Female"])
age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)   
balance = st.number_input("Enter Balance", min_value=0.0, value=1000.0)
credit_score = st.number_input("Enter Credit Score", min_value=300, max_value=850, value=600)
estimated_salary = st.number_input("Enter Estimated Salary", min_value=0.0, value=50000.0)
tenure = st.number_input("Enter Tenure", min_value=0, max_value=10, value=3)
num_of_products = st.number_input("Enter Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])



#prepare input data for prediction
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],   
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == "Yes" else 0],
    'IsActiveMember': [1 if is_active_member == "Yes" else 0],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))


input_data  = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1) 

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
pred_prob = prediction[0][0]

if pred_prob > 0.5:
    st.write(f"The customer is likely to churn with a probability of {pred_prob:.2f}")    
else:
    st.write(f"The customer is unlikely to churn with a probability of {pred_prob:.2f}")