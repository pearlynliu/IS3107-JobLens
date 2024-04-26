import streamlit as st
from prediction import load_model
import numpy as np
import pandas as pd


# Page title
st.set_page_config(page_title='JobLens', page_icon='🤖')
st.title('💰 Salary Predictor')

df = pd.read_csv('combined_v4.csv')

model, label_encoder, vectorizer, lda, scaler = load_model()

# Create input fields
job_type = st.selectbox('Job Type', ['Full time', 'Contract', 'Unknown'])  # Add actual job types used in training
job_field = st.selectbox('Job Field', ['Data Analyst', 'Data Scientist', 'Data Engineer', 'Data Architect', 'Database Administrator', 'Software Engineer', 'Others'])  # Add actual job fields used in training
job_description = st.text_area('Job Description')

# Dummy function to mimic the original preprocessing, adjust as needed
def preprocess_inputs(job_type, job_field, job_description):
    # These should only transform based on previously fitted state.
    job_type_dict = {
        'Contract': 0,
        'Full time': 1,
        'Unknown': 2
    }

    job_field_dict = {
        'Others': 0,
        'data analyst': 1,
        'data architect': 2,
        'data engineer': 3,
        'data scientist': 4,
        'database administrator': 5,
        'software engineer': 6
    }

    job_type_encoded = job_type_dict[job_type]
    job_field_encoded = job_field_dict[job_field.lower()]
    job_description_vectorized = vectorizer.transform([job_description])
    
    # Assuming lda was saved and loaded similarly:
    lda_features = lda.transform(job_description_vectorized)

    Is_internship = 0
    priority = 1
    # Combine into a single sample for prediction
    features = np.hstack([Is_internship, priority, job_type_encoded, job_field_encoded, np.argmax(lda_features, axis=1)])

    return features

if st.button('Predict Salary'):
    # Preprocess the inputs
    features = preprocess_inputs(job_type, job_field, job_description)
    
    # Predict salary
    predicted_salary_scaled = model.predict([features])[0]  # Predict method expects an array of samples
    predicted_salary = scaler.inverse_transform([[predicted_salary_scaled]])
    
    # Display the predicted salary
    st.write(f'The predicted salary is: ${predicted_salary[0][0]:.2f}')