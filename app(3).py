import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set page configuration for a wider layout and modern look
st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

# Custom CSS for modern styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .stTextInput, .stNumberInput, .stSelectbox {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load the pre-trained model
try:
    model = joblib.load('loan_model.pkl')
except FileNotFoundError:
    st.error("Model file 'loan_model.pkl' not found. Please ensure the model is saved in the same directory as this app.")
    st.stop()

# Function to preprocess input data
def preprocess_input(data):
    # Map categorical variables to match training data encoding
    gender_map = {'Male': 1, 'Female': 0, 'Other': 2}
    married_map = {'Yes': 1, 'No': 0}
    education_map = {'Graduate': 1, 'Not Graduate': 0}
    self_employed_map = {'Yes': 1, 'No': 0}
    property_area_map = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
    dependents_map = {'0': 0, '1': 1, '2': 2, '3+': 3}

    # Create a DataFrame from input
    input_df = pd.DataFrame([data])
    
    # Apply mappings
    input_df['Gender'] = input_df['Gender'].map(gender_map)
    input_df['Married'] = input_df['Married'].map(married_map)
    input_df['Education'] = input_df['Education'].map(education_map)
    input_df['Self_Employed'] = input_df['Self_Employed'].map(self_employed_map)
    input_df['Property_Area'] = input_df['Property_Area'].map(property_area_map)
    input_df['Dependents'] = input_df['Dependents'].map(dependents_map)
    
    # Ensure all expected columns are present and in correct order
    expected_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                       'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                       'Loan_Amount_Term', 'Credit_History', 'Property_Area']
    input_df = input_df[expected_columns]
    
    return input_df

# Sidebar for user input
st.sidebar.header("Loan Applicant Details")
with st.sidebar.form(key='loan_form'):
    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    married = st.selectbox("Married", ['Yes', 'No'])
    dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
    education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
    applicant_income = st.number_input("Applicant Income ($)", min_value=0.0, step=100.0)
    coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0.0, step=100.0)
    loan_amount = st.number_input("Loan Amount ($ thousands)", min_value=0.0, step=1.0)
    loan_term = st.number_input("Loan Term (months)", min_value=0.0, step=12.0)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])
    submit_button = st.form_submit_button(label="Predict Loan Status")

# Main content
st.title("Loan Approval Predictor")
st.markdown("Enter the applicant's details in the sidebar to predict loan approval status.")

if submit_button:
    # Collect input data
    input_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }
    
    # Preprocess input
    processed_input = preprocess_input(input_data)
    
    # Make prediction
    prediction = model.predict(processed_input)[0]
    prediction_proba = model.predict_proba(processed_input)[0]
    
    # Display prediction
    st.subheader("Prediction Result")
    status = "Approved" if prediction == 1 else "Rejected"
    st.markdown(f"**Loan Status:** {status}")
    st.markdown(f"**Approval Probability:** {prediction_proba[1]:.2%}")
    
    # Summary of inputs
    st.subheader("Input Summary")
    input_summary = pd.DataFrame([input_data]).T.rename(columns={0: 'Value'})
    st.table(input_summary)
    
    # Visualizations
    st.subheader("Model Insights")
    
    # Feature Importance
    st.markdown("**Feature Importance**")
    feature_names = processed_input.columns
    coefficients = model.coef_[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=coefficients, y=feature_names, ax=ax)
    ax.set_title("Feature Importance (Logistic Regression Coefficients)")
    ax.set_xlabel("Coefficient Value")
    st.pyplot(fig)
    
    # Confusion Matrix (using sample data for demonstration)
    st.markdown("**Confusion Matrix**")
    # For demo, assume we have some sample true and predicted labels
    # In a real app, you'd need test data; here we simulate
    sample_y_true = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]  # Example true labels
    sample_y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1]  # Example predicted labels
    cm = confusion_matrix(sample_y_true, sample_y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | © 2025 Loan Approval Predictor")