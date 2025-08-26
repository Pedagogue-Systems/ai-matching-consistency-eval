import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

def display_results(results):
    st.dataframe(results)

def model_flag_rate(results, model):
    df = results[results['model_name'] == model]

    hard_flag = len(df[df['flag'] == 2])
    soft_flag = len(df[df['flag'] == 1])
    no_flag = len(df[df['flag'] == 0])

    flag = [hard_flag, soft_flag, no_flag]

    labels = ['Hard Flag', 'Soft Flag', 'No Flag']
    colors = ['crimson', 'indigo', 'green']

    fig, ax = plt.subplots()
    ax.pie(flag, labels=labels, colors=colors, autopct='%1.1f%%')
    ax.set_title('Altered Resume Flag Rate')
    st.pyplot(fig)

def top_job_shift(results, resume, model):
    df = results[results['resume_A'] == resume]
    df = df[df['model_name'] == model]

    baseline = df.sort_values(by='score_baseline', ascending=False).head(10)
    variant = df.sort_values(by='score_variant', ascending=False)

    baseline_top = baseline['job_posting'].tolist()
    variant_top = variant['job_posting'].tolist()

    table = pd.DataFrame(columns=['Job ID', 'Initial Position', 'New Position', 'Change'])

    for i, job in enumerate(baseline_top):
        new_pos = variant_top.index(job)
        change = (i+1) - (new_pos+1)
        row = {'Job ID': job,
               'Initial Position': i+1,
               'New Position': new_pos+1,
               'Change': f'{change:+}'}
        table.loc[len(table)] = row

    st.table(table)

def top_job_shift_all(results, resume):
    model_names = results['model_name'].unique().tolist()
    for model in model_names:
        st.subheader(model)
        top_job_shift(results, resume, model)

def top_candidate_shift(results, job, model):
    df = results[results['job_posting'] == job]
    df = df[df['model_name'] == model]

    baseline = df.sort_values(by='score_baseline', ascending=False).head(10)
    variant = df.sort_values(by='score_variant', ascending=False)

    baseline_top = baseline['resume_A'].tolist()
    variant_top = variant['resume_A_prime'].tolist()

    table = pd.DataFrame(columns=['Resume ID', 'Initial Position', 'New Position', 'Change'])

    for i, resume in enumerate(baseline_top):
        new_pos = variant_top.index(f'{resume}_prime')
        change = (i+1) - (new_pos+1)
        row = {'Resume ID': resume,
                'Initial Position': i+1,
                'New Position': new_pos+1,
                'Change': f'{change:+}'}
        table.loc[len(table)] = row

    st.table(table)

def top_candidate_shift_all(results, job):
    model_names = results['model_name'].unique().tolist()
    for model in model_names:
        st.subheader(model)
        top_candidate_shift(results, job, model)


if __name__ == '__main__':
    results = pd.read_csv('data/results.csv')

    job_ids = results['job_posting'].unique().tolist()
    resume_ids = results['resume_A'].unique().tolist()
    model_names = results['model_name'].unique().tolist()

    st.title('AI Matching Consistency Evaluation')

    job_option = st.selectbox('Job ID', job_ids, index=None, placeholder='All')
    resume_option = st.selectbox('Resume ID', resume_ids, index=None, placeholder='All')
    model_option = st.selectbox('Model', model_names, index=None, placeholder='All')

    if st.button('Run', width='stretch'):
        if job_option is None and resume_option is None and model_option is None:
            display_results(results)
        elif job_option is None and resume_option is None and model_option is not None:
            model_flag_rate(results, model_option)
        elif job_option is None and resume_option is not None and model_option is None:
            top_job_shift_all(results, resume_option)
        elif job_option is None and resume_option is not None and model_option is not None:
            top_job_shift(results, resume_option, model_option)
        elif job_option is not None and resume_option is None and model_option is None:
            top_candidate_shift_all(results, job_option)
        elif job_option is not None and resume_option is None and model_option is not None:
            top_candidate_shift(results, job_option, model_option)
        elif job_option is not None and resume_option is not None and model_option is None:
            pass
        elif job_option is not None and resume_option is not None and model_option is not None:
            pass
