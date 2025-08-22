import pandas as pd
import streamlit as st

results = pd.read_csv('data/results.csv')

job_ids = results['job_posting'].unique().tolist()
resume_ids = results['resume_A'].unique().tolist()
model_names = results['model_name'].unique().tolist()

st.title('AI Matching Consistency Evaluation')

job_option = st.selectbox(
    'Job ID',
    job_ids,
    index=None,
    placeholder='None'
)

resume_option = st.selectbox(
    'Resume ID',
    resume_ids,
    index=None,
    placeholder='None'
)

model_option = st.selectbox(
    'Model',
    model_names,
    index=None,
    placeholder='None'
)

if st.button('Run', width='stretch'):
    st.write(f'{job_option}, {resume_option}, {model_option}')
