import streamlit as st
import os
from google.oauth2 import service_account
import pandas_gbq
import numpy as np

# Setup credentials
def get_credentials():
    parent_wd = os.path.dirname(os.getcwd())
    cred_path = os.path.join(parent_wd,  "auth", "is3107-416813-f8b1bf76ef57.json")
    return service_account.Credentials.from_service_account_file(cred_path)

credentials = get_credentials()
project_id = "is3107-416813"
final_id = 'is3107_scraped_data.final_table'

# SQL query
sql = f"SELECT * FROM `{project_id}.{final_id}`"

@st.cache_data
def load_data():
    """Load data from BigQuery, drop duplicates, and cache it for future use across scripts."""
    df = pandas_gbq.read_gbq(sql, project_id=project_id, credentials=credentials)
    
    # Drop duplicates based on 'Title', 'Company'
    df_unique = df.drop_duplicates(subset=['Title', 'Company'])

    # Filter out rows where 'Company' is a string 'null'
    df_clean = df_unique[df_unique['Company'].str.lower() != 'null']
    
    
    return df_clean


