import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from data_loader import load_data

# Page title
st.set_page_config(page_title='JobLens', page_icon='🤖')
st.title('📈 Job Trends')

########## Load dataset ##########

df = load_data()

########## Dashboard ##########

##### Most Popular Roles in Past 7 Days #####
# Convert date column to datetime
try:
    df['Date_scraped'] = pd.to_datetime(df['Date_scraped'], format='%Y/%m/%d')
except Exception as e:
    st.error(f"Error converting date column: {e}")
    st.stop()

# Ensure consistency of fields
df['Field'] = df['Field'].str.lower().str.strip("{''}").str.title()

# Replace empty field with NaN
df['Field'] = df['Field'].replace('', None)

# handling 'Field' -- since it only has 21 nans, we will impute with the most common title, which is software engineer
df['Field'] = df['Field'].fillna('Software Engineer')

df['Field'] = df['Field'].replace('Data', 'Data Analyst')


# Filter data for the past 7 days
end_date = datetime.today()
start_date = end_date - timedelta(days=7)

try:
    filtered_data = df[(df['Date_scraped'] >= start_date) & (df['Date_scraped'] <= end_date)]
except Exception as e:
    st.error(f"Error filtering data: {e}")
    st.stop()

# Count occurrences of each role
role_counts = filtered_data['Field'].value_counts()

# Select top 10 most popular roles
top_10_roles = role_counts.head(10).reset_index()
top_10_roles.columns = ['role', 'count']

# Calculate percentage for each role
top_10_roles['percentage'] = (top_10_roles['count'] / top_10_roles['count'].sum()) * 100

# Plot top 10 roles as a pie chart with percentages using Plotly
fig = px.pie(top_10_roles, values='count', names='role', title='Proportion of Roles',
             labels={'role':'Role', 'count':'Count'},
             hover_data=['percentage'], hole=0.3)

# Display the pie chart
st.plotly_chart(fig)


##### Salary Comparison over Various Roles #####
# Remove rows with empty salary data
drop_na_salary = filtered_data.dropna(subset=['Salary_min_month', 'Salary_max_month'])

# Group by role and calculate mean salary
salary_comparison = drop_na_salary.groupby('Field').agg({'Salary_min_month': 'mean', 'Salary_max_month': 'mean'}).reset_index()

# Plot salary comparison for top 10 roles using Plotly
fig = px.bar(salary_comparison, x='Field', y=['Salary_min_month', 'Salary_max_month'], 
             barmode='group', 
             labels={'Field': 'Role', 'value': 'Salary per month (SGD)', 'variable': 'Salary Type'},
             title='Comparison of Minimum and Maximum Salaries')

# Change legend labels
fig.update_traces(
    name='Minimum Salary', 
    selector=dict(name='Salary_min_month')
)

fig.update_traces(
    name='Maximum Salary', 
    selector=dict(name='Salary_max_month')
)

fig.update_layout(xaxis={'categoryorder': 'total descending'})  # Sort x-axis categories by total salary
st.plotly_chart(fig)


##### Job Postings Over Time #####
# Group data by Date and count number of job postings
time_series_data = df.groupby('Date_scraped')['Title'].count().reset_index()
time_series_data.columns = ['Date', 'Number of Job Postings']

# Plot time series data using Plotly
fig = px.line(time_series_data, x='Date', y='Number of Job Postings', 
              title='Number of Job Postings Over Time',
              labels={'Date': 'Date', 'Number of Job Postings': 'Number of Job Postings'})
st.plotly_chart(fig)


##### Distribution of Job Postings by Company #####
# Count occurrences of each company
company_data = filtered_data.dropna(subset=['Company'])
company_data['Company'] = company_data['Company'].replace('TIKTOK PTE. LTD.', 'TikTok')
company_data['Company'] = company_data['Company'].replace('IMD Info-communications Media Development Authority', 'IMDA')
company_counts = company_data['Company'].value_counts()

# Select top 10 companies
top_10_companies = company_counts.head(10).reset_index()
top_10_companies.columns = ['Company', 'Count']

# Plot top 10 companies as a bar chart using Plotly
fig = px.bar(top_10_companies, x='Company', y='Count',
             title='Top 10 Companies Posting the Most in the Past 7 Days',
             labels={'Count': 'Number of Posts', 'Company': 'Company Name'},
             color='Company',  # This will color each bar differently
             template='plotly_white')

# Adjust layout
fig.update_layout(xaxis_title="Company Name",
                  yaxis_title="Number of Posts",
                  xaxis_tickangle=-45)

# Display the chart in Streamlit
st.plotly_chart(fig)
