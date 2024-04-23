import streamlit as st
from data_loader import load_data

# Page title
st.set_page_config(page_title='JobLens', page_icon='🤖')
st.title('💰 Salary Predictor')

df = load_data()



