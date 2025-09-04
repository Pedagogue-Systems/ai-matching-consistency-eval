import re, json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

def display_results(results):
    st.dataframe(results)

def model_statistics(results, model):
    df = results[results['model_name'] == model]

    stats = df.groupby('model_name')['delta'].agg(['var', 'std', 'mean', 'min', 'max'])
    st.header('Altered Resume Statistics')
    st.table(stats)

    hard_flag = len(df[df['flag'] == 2])
    soft_flag = len(df[df['flag'] == 1])
    no_flag = len(df[df['flag'] == 0])
    total = hard_flag + soft_flag + no_flag

    flag = [hard_flag, soft_flag, no_flag]
    labels = [f'Hard Flag ({(hard_flag / total) * 100:.2f}%)',
              f'Soft Flag ({(soft_flag / total) * 100:.2f}%)',
              f'No Flag ({(no_flag / total) * 100:.2f}%)']
    colors = ['crimson', 'indigo', 'green']

    st.subheader('')
    st.header('Altered Resume Flag Rate')
    fig, ax = plt.subplots()
    ax.pie(flag, labels=labels, colors=colors)

    for text in fig.findobj(match=plt.Text):
        text.set_color('white')

    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')

    st.pyplot(fig, transparent=True)

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

def read_files(resume_path, job_postings_path):
    resumes = []
    with open(resume_path, 'r') as file:
        for line in file:
            resumes.append(line.strip())

    df = pd.read_csv(job_postings_path)
    job_postings = df['job_description'].tolist()

    return resumes, job_postings

def get_resume(resumes, index):
    resume = resumes[index]
    resume_json = json.loads(resume)
    resume = json.dumps(resume_json, indent=4)
    label = fr'$\Large \textsf{{resume\_{index}}}$'
    st.text_area(label, resume, height=300)

def get_resume_prime(resumes, index):
    resume = resumes[index]
    
    resume_prime = ''
    data = json.loads(resume)
    if 'personal_info' in data and 'summary' in data['personal_info']:
        summary = data['personal_info']['summary']
        sentences = re.split(r'(?<=[.!?]) +', summary)
        data['personal_info']['summary'] = ' '.join(sentences[1:])
        resume_prime = json.dumps(data, indent=4)

    label = fr'$\Large \textsf{{resume\_{index}\_prime}}$'
    st.text_area(label, resume_prime, height=300)

def get_job_posting(job_postings, index):
    job_posting = job_postings[index]
    label = fr'$\Large \textsf{{job\_{index}}}$'
    st.text_area(label, job_posting, height=300)

def resume_to_job_scoring(results, job_postings, resumes, job, resume, model):
    df = results[(results['job_posting'] == job) & (results['model_name'] == model)]
    
    baseline = df.sort_values(by='score_baseline', ascending=False)
    variant = df.sort_values(by='score_variant', ascending=False)

    score_baseline = df[df['resume_A'] == resume]['score_baseline'].iloc[0]
    rank_baseline = baseline['resume_A'].tolist().index(resume) + 1
    row_baseline = {'Resume ID': resume,
                    'Score': score_baseline,
                    'Rank': rank_baseline}

    score_variant = df[df['resume_A_prime'] == f'{resume}_prime']['score_variant'].iloc[0]
    rank_variant = variant['resume_A_prime'].tolist().index(f'{resume}_prime') + 1
    row_variant = {'Resume ID': f'{resume}_prime',
                    'Score': score_variant,
                    'Rank': rank_variant}

    table = pd.DataFrame(columns=['Resume ID', 'Score', 'Rank'])
    table.loc[len(table)] = row_baseline
    table.loc[len(table)] = row_variant

    st.header('Resume-to-Job Scoring')
    st.table(table)

    resume_ids = [resume, f'{resume}_prime']
    scores = [score_baseline, score_variant]
    colors = ['indigo', 'crimson']

    fig, ax = plt.subplots()
    ax.bar(resume_ids, scores, color=colors)
    ax.set_ylabel('Similarity Score')
    
    for text in fig.findobj(match=plt.Text):
        text.set_color('white')

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    
    st.pyplot(fig, transparent=True)

    resume_index = int(resume_option.split('_')[1])
    job_index = int(job_option.split('_')[1])

    get_resume(resumes, resume_index)
    get_resume_prime(resumes, resume_index)
    get_job_posting(job_postings, job_index)

def resume_to_job_scoring_all(results, job_postings, resumes, job, resume):
    model_names = results['model_name'].unique().tolist()

    df = results[results['job_posting'] == job]

    for model in model_names:
        st.subheader(model)

        model_df = df[df['model_name'] == model]

        baseline = model_df.sort_values(by='score_baseline', ascending=False)
        variant = model_df.sort_values(by='score_variant', ascending=False)

        score_baseline = model_df[model_df['resume_A'] == resume]['score_baseline'].iloc[0]
        rank_baseline = baseline['resume_A'].tolist().index(resume) + 1
        row_baseline = {'Resume ID': resume,
                        'Score': score_baseline,
                        'Rank': rank_baseline}

        score_variant = model_df[model_df['resume_A_prime'] == f'{resume}_prime']['score_variant'].iloc[0]
        rank_variant = variant['resume_A_prime'].tolist().index(f'{resume}_prime') + 1
        row_variant = {'Resume ID': f'{resume}_prime',
                        'Score': score_variant,
                        'Rank': rank_variant}

        table = pd.DataFrame(columns=['Resume ID', 'Score', 'Rank'])
        table.loc[len(table)] = row_baseline
        table.loc[len(table)] = row_variant

        st.table(table)

    resume_index = int(resume_option.split('_')[1])
    job_index = int(job_option.split('_')[1])

    get_resume(resumes, resume_index)
    get_resume_prime(resumes, resume_index)
    get_job_posting(job_postings, job_index)


if __name__ == '__main__':
    resume_path = 'data/resumes/master_resumes.jsonl'
    job_postings_path = 'data/job_postings/training_data.csv'
    resumes, job_postings = read_files(resume_path, job_postings_path)

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
            model_statistics(results, model_option)
        elif job_option is None and resume_option is not None and model_option is None:
            top_job_shift_all(results, resume_option)
        elif job_option is None and resume_option is not None and model_option is not None:
            top_job_shift(results, resume_option, model_option)
        elif job_option is not None and resume_option is None and model_option is None:
            top_candidate_shift_all(results, job_option)
        elif job_option is not None and resume_option is None and model_option is not None:
            top_candidate_shift(results, job_option, model_option)
        elif job_option is not None and resume_option is not None and model_option is None:
            resume_to_job_scoring_all(results, job_postings, resumes, job_option, resume_option)
        elif job_option is not None and resume_option is not None and model_option is not None:
            resume_to_job_scoring(results, job_postings, resumes, job_option, resume_option, model_option)
