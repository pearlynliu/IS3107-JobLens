import streamlit as st
import os
from google.oauth2 import service_account
import pandas_gbq
import numpy as np

# Setup credentials
def get_credentials():
    creds = st.secrets["gcp_service_account"]
    return service_account.Credentials.from_service_account_info(creds)

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

    # re-binning for 'Field'
    df_clean['Field'] = np.where(
        df_clean['Field'].str.contains('analyst', case=False), 'data analyst',
        np.where(df_clean['Field'].str.contains('scientist', case=False), 'data scientist',
            np.where((df_clean['Field'].str.contains('data engineer', case=False)) | (df_clean['Field'].str.contains('machine learning engineer', case=False)), 'data engineer',
            np.where(df_clean['Field'].str.contains('architect', case=False), 'data architect',
                np.where(df_clean['Field'].str.contains('database', case=False), 'database administrator',
                np.where((df_clean['Field'].str.contains('software engineer', case=False)) | (df_clean['Field'].str.contains('developer', case=False)), 'software engineer', 'Others')
                    # while 'Others' may seem very vague, from our list of filtered keywords, we know that these are all still data-related jobs.
    )))))
    
    return df_clean


