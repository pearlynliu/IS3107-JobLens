import streamlit as st
import pdfplumber
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from itertools import chain
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from data_loader import load_data

# Page title
st.set_page_config(page_title='JobLens', page_icon='🤖')
st.title('💼 Job Recommender')

# Set the NLTK data path to use local resources
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))


########## HELPER FUNCTIONS ##########
def extract_text_from_pdf(uploaded_file):
    text = ""
    # Check if the uploaded file is a string path (when shared) or a file-like object (when local)
    if isinstance(uploaded_file, str):
        # When running with share=True, Gradio provides the file path as a string
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ''  # Fallback for pages with no text
    else:
        # If uploaded_file is a file-like object (running locally without sharing)
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ''
    return text

# Split pdf_text into shorter substrings first so that the splicing function works more effectively
def split_into_paragraphs(document):
    final_list = []
    ongoing_word = ""
    num_words_counter = 0
    for char in document:
        if char.isspace():
            num_words_counter += 1
        ongoing_word += char
        if num_words_counter == 30:
            final_list.append(ongoing_word)
            ongoing_word = ""
            num_words_counter = 0
    if ongoing_word:
        final_list.append(ongoing_word)
    return final_list

def regex_process_text(text):
    processed_text = " "
    text_lower = text.lower()
    # tokenize (keep alphabetical only)
    tokenizer = RegexpTokenizer(r"[A-Za-z']+[A-Za-z]|[A-Za-z]+")
    tokens = tokenizer.tokenize(text_lower)
    # define stopwords, removing some
    english_stopwords = stopwords.words('english')
    stopwords_to_keep = ['not', 'no', "don't", "won't", "didn't", "but", "weren't", "wouldn't"]
    for i in stopwords_to_keep:
        english_stopwords.remove(i)
    tokens = [word for word in tokens if word not in english_stopwords]
    # lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    processed_text = processed_text.join(tokens)
    return processed_text

# processing resume which has been split into multiple chunks -- improves performance
def splitting_tokenizing_lemmatizing(paragraph_list):
  tokenized_paragraph_list = []
  for paragraph in paragraph_list:
    tokens = regex_process_text(paragraph)
    tokens_nltk = tokens.split()
    tokenized_paragraph_list.append(tokens_nltk)
  return tokenized_paragraph_list

# same as splitting_tokenizing_lemmatizingm, but for dataframe application
def splitting_tokenizing_lemmatizing_individual(paragraph):
  tokens = regex_process_text(paragraph)
  tokens_nltk = tokens.split()
  return tokens_nltk

def compute_cosine_similarity(list1, list2): # similarity between resume + job desc
    str1 = ' '.join(list1)
    str2 = ' '.join(list2)
    tokenized_list1 = str1.split()
    tokenized_list2 = str2.split()
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([str1, str2])
    cosine_similarity_score = cosine_similarity(X[0], X[1])[0][0]
    return cosine_similarity_score


########## MAIN LOGIC ##########

resume_file = st.file_uploader("Upload your resume", type=['pdf', 'docx'])
jobs = load_data()

if resume_file and not jobs.empty:
    # Process resume
    resume_text = extract_text_from_pdf(resume_file)
    resume_paragraphs = split_into_paragraphs(resume_text)
    split_and_tokenized_resume_list = splitting_tokenizing_lemmatizing(resume_paragraphs)
    # flatmap to combine all chunks into 1 list
    flattened_resume_list = list(chain(*split_and_tokenized_resume_list))

    # Process job descriptions
    jobs['processed'] = jobs['Description'].apply(lambda x: splitting_tokenizing_lemmatizing_individual(x))
    jobs['cosine_similarity'] = jobs['processed'].apply(lambda x: compute_cosine_similarity(x, flattened_resume_list))

    scaler = MinMaxScaler()
    jobs['scaled_cosine_similarity'] = scaler.fit_transform(jobs[['cosine_similarity']])
    jobs_sorted = jobs.sort_values(by='scaled_cosine_similarity', ascending=False)

    user_view_jobs = jobs_sorted[['Title', 'Company', 'Application_link', 'Salary_min_month', 'Salary_max_month']].head(10) # just show top 10 most suitable listings
    user_view_jobs['job_rank'] = range(1, len(user_view_jobs) + 1)

    st.subheader("Top 10 Job Matches:")
    for index, row in user_view_jobs.iterrows():
        rank = row['job_rank']
        title = row['Title']
        company = row['Company']
        application_link = row['Application_link']

        # Create two columns: one for the text and one for the button
        col1, col2 = st.columns([4, 1])  # Adjust the ratio as per your requirement
 
        # Use the first column to display the text
        col1.markdown(f"{rank}. {company} - {title}")

        # Use the second column to display the button
        col2.link_button("Apply", application_link)

