import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load


df = pd.read_csv('combined_v4.csv')

########## DATA PREPARATION ##########
df_clean = df.copy()


###### HANDLE NANS ######
# get rid of cols that are irrelevant to salary prediction
df_clean = df_clean.drop(['Created_date', 'Job_id', 'Founded', 'Revenue', 'Application_link'], axis=1) # revenue and founded corresponds to the company anyway, not needed

# we cannot make salary predictions on jobs that don't even have salary data. hence, drop rows without salary data
df_clean = df_clean.dropna(subset=['Salary_min_month'])

# drop original salary col
df_clean = df_clean.drop(['Salary'], axis=1)

# handle nans in non-numeric cols
non_numeric_cols = ['Company', 'Requirements_short', 'Requirements_full', 'Type',
       'Responsibilities', 'Location', 'Industry', 'Sector', 'Availability_requests', 'Size']
for i in non_numeric_cols:
  df_clean[i] = df_clean[i].fillna('Unknown') # since the information is unknown, we should literally fill nans up with "Unknown"

# handling 'Field' -- since it only has 21 nans, we will impute with the most common title, which is software engineer
df_clean['Field'] = df_clean['Field'].fillna('Software engineer')

# drop Title, bcos Field already contains the parsed form
df_clean = df_clean.drop(['Title'], axis=1)

# handle nans in numeric cols
df_clean['Is_internship'] = df_clean['Is_internship'].fillna(0)

##### HANDLE DTYPES #####
df_clean2 = df_clean.copy()

# Drop Data_source -- irrelevant
df_clean2 = df_clean2.drop(['Data_source'], axis=1)
df_clean2 = df_clean2.drop(['Unnamed: 0'], axis=1)

df_clean2['Date_scraped'] = pd.to_datetime(df_clean2['Date_scraped'])
df_clean2['Field'] = df_clean2['Field'].str.replace("[^a-zA-Z\s]", "", regex=True)

# since description is derived from requirements, drop requirements col
df_clean2 = df_clean2.drop(['Requirements_short', 'Requirements_full'], axis=1)


##### HANDLE CATEGORICAL VARIABLES #####
'''
handling the binning

for highly imbalanced categories, it is not meaningful to keep these columns, since the values should be the same for almost the whole dataset
in some cases, e.g. Location, it is indeed the same Location for the whole dataset (Singapore). But due to recording differences between different platforms, there are different categories showing up
'''
# drop these categories
df_clean2 = df_clean2.drop(['Availability_requests', 'Sector', 'Location', 'Size', 'Industry', 'Company'], axis=1)

# re-binning for 'Type'
df_clean2['Job_Type'] = np.where(
    df_clean2['Type'].str.contains('Full'), 'Full time',
      np.where(df_clean2['Type'].str.contains('Perm'), 'Contract',
        np.where(df_clean2['Type'].str.contains('Unknown'), 'Unknown',
          np.where(df_clean2['Type'].str.contains('Fresh graduate'), 'Full time',
            'Contract'))))

# drop original 'Type' column
df_clean2 = df_clean2.drop(['Type'], axis=1)

# re-binning for 'Field'
df_clean2['Job_Field'] = np.where(
    df_clean2['Field'].str.contains('analyst', case=False), 'data analyst',
      np.where(df_clean2['Field'].str.contains('scientist', case=False), 'data scientist',
        np.where((df_clean2['Field'].str.contains('data engineer', case=False)) | (df_clean2['Field'].str.contains('machine learning engineer', case=False)), 'data engineer',
          np.where(df_clean2['Field'].str.contains('architect', case=False), 'data architect',
            np.where(df_clean2['Field'].str.contains('database', case=False), 'database administrator',
              np.where((df_clean2['Field'].str.contains('software engineer', case=False)) | (df_clean2['Field'].str.contains('developer', case=False)), 'software engineer', 'Others')
                # while 'Others' may seem very vague, from our list of filtered keywords, we know that these are all still data-related jobs.
)))))

# drop original 'Field' column
df_clean2 = df_clean2.drop(['Field'], axis=1)

# handling 'Description'
# Vectorize the Text
vectorizer = CountVectorizer(stop_words='english')
dtm = vectorizer.fit_transform(df_clean2['Description'])  # Document-term matrix

# Fit LDA Model
lda = LatentDirichletAllocation(n_components=10, random_state=0)  # Assuming we want to find 10 topics
lda_features = lda.fit_transform(dtm)

# just take the highest probability topic as a feature for simplicity
df_clean2['Dominant_Topic'] = np.argmax(lda_features, axis=1)

# drop original Description column
df_clean2 = df_clean2.drop(['Description'], axis=1)

# drop Responsibities
df_clean2 = df_clean2.drop(['Responsibilities'], axis=1)

# create a new salary column, which is the midpoint of salary_min and salary_max, so that we only have 1 variable to predict
df_clean2['Salary_midpoint'] = (df_clean2['Salary_min_month'] + df_clean2['Salary_max_month']) / 2

# drop original salary columns
df_clean2 = df_clean2.drop(['Salary_min_month', 'Salary_max_month'], axis=1)

# drop date column -- not doing time series
df_clean2 = df_clean2.drop(['Date_scraped'], axis=1)

df_clean2['Is_internship'] = df_clean2['Is_internship'].astype('int')


########## MACHINE LEARNING ##########
label_encoder = LabelEncoder()
df_clean2['Job_Type'] = label_encoder.fit_transform(df_clean2['Job_Type'])
df_clean2['Job_Field'] = label_encoder.fit_transform(df_clean2['Job_Field'])

##### Remove Outliers #####
Q1 = df_clean2['Salary_midpoint'].quantile(0.25)
Q3 = df_clean2['Salary_midpoint'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_clean_no_outliers = df_clean2[(df_clean2['Salary_midpoint'] >= lower_bound) & (df_clean2['Salary_midpoint'] <= upper_bound)]

# drop remaining columns that are irrelevant
# df_clean_no_outliers = df_clean_no_outliers.drop(['Is_internship', 'priority'], axis=1)

# scaling target
X = df_clean_no_outliers.drop(['Salary_midpoint'], axis=1)
y = df_clean_no_outliers['Salary_midpoint']

scaler = MinMaxScaler(feature_range=(0, 1))
y = scaler.fit_transform(y.values.reshape(-1, 1))

# train test split -- no need val since we are only using 1 model and our dataset so small
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

random_forest_regressor = RandomForestRegressor(
                              n_estimators=700,
                              max_depth=10,
                              max_features='log2',
                              warm_start=True,
                              random_state=42)
random_forest_regressor.fit(X_train, y_train)

# Make predictions for test set
y_pred = random_forest_regressor.predict(X_test)

########## SAVE THE MODEL ##########
dump(random_forest_regressor, 'random_forest_regressor.joblib')
dump(label_encoder, 'label_encoder.joblib')
dump(vectorizer, 'vectorizer.joblib')
dump(lda, 'lda.joblib')
dump(scaler, 'scaler.joblib')

@st.cache_resource
def load_model():
    model = load('random_forest_regressor.joblib')
    label_encoder = load('label_encoder.joblib')
    vectorizer = load('vectorizer.joblib')
    lda = load('lda.joblib')
    scaler = load('scaler.joblib')
    return model, label_encoder, vectorizer, lda, scaler

