import re, json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

def display_results(results):
    st.header('Results')
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

    st.pyplot(fig, transparent=True)

def top_job_overlap(results, resume):
    df = results[results['resume_A'] == resume]
    model_names = results['model_name'].unique().tolist()

    model_0 = df[df['model_name'] == model_names[0]]
    model_0 = model_0.sort_values(by='score_baseline', ascending=False).head(10)
    model_1 = df[df['model_name'] == model_names[1]]
    model_1 = model_1.sort_values(by='score_baseline', ascending=False).head(10)
    model_2 = df[df['model_name'] == model_names[2]]
    model_2 = model_2.sort_values(by='score_baseline', ascending=False).head(10)

    r0 = set(model_0['job_posting'])
    r1 = set(model_1['job_posting'])
    r2 = set(model_2['job_posting'])

    st.header('Top Job Overlap')

    labels = ('Model 0', 'Model 1', 'Model 2')
    colors = ('mediumslateblue', 'hotpink', 'lime')

    fig, ax = plt.subplots()
    venn = venn3([r0, r1, r2], set_labels=labels, set_colors=colors)

    for label in venn.set_labels:
        label.set_color('white')

    st.pyplot(fig, transparent=True)

    st.markdown('***')
    st.write(f'{labels[0]} -- {model_names[0]}')
    st.write(f'{labels[1]} -- {model_names[1]}')
    st.write(f'{labels[2]} -- {model_names[2]}')

def top_job_shift(results, resume, model):
    df = results[results['resume_A'] == resume]
    df = df[df['model_name'] == model]

    baseline = df.sort_values(by='score_baseline', ascending=False).head(10)
    variant = df.sort_values(by='score_variant', ascending=False)

    baseline_top = baseline['job_posting'].tolist()
    variant_top = variant['job_posting'].tolist()

    table = pd.DataFrame(columns=['Job ID', 'Initial Position', 'New Position', 'Change'])

    job_scores = {}

    for i, job in enumerate(baseline_top):
        new_pos = variant_top.index(job)
        change = (i+1) - (new_pos+1)
        row = {'Job ID': job,
               'Initial Position': i+1,
               'New Position': new_pos+1,
               'Change': f'{change:+}'}
        table.loc[len(table)] = row

        score_baseline = df[df['job_posting'] == job]['score_baseline'].iloc[0]
        score_variant = df[df['job_posting'] == job]['score_variant'].iloc[0]
        job_scores[job] = (score_baseline, score_variant)

    st.header('Top Job Shift')
    st.table(table)

    job_names = table['Job ID'].tolist()
    plot = pd.DataFrame(job_scores, index=['baseline', 'variant']).T

    x = np.arange(10)
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, plot['baseline'], width, label=resume, color='indigo')
    ax.bar(x + width/2, plot['variant'], width, label=f'{resume}_prime', color='crimson')
    ax.set_ylabel('Similarity Score')
    ax.set_xlabel('Job ID')
    ax.set_xticks(x)
    ax.set_xticklabels(job_names)
    ax.tick_params(axis='x', labelrotation=45)
    ax.legend(frameon=False)

    for text in fig.findobj(match=plt.Text):
        text.set_color('white')

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    st.pyplot(fig, transparent=True)

def top_job_shift_all(results, resume):
    model_names = results['model_name'].unique().tolist()
    for model in model_names:
        st.subheader(model)
        top_job_shift(results, resume, model)

def top_candidate_overlap(results, job):
    df = results[results['job_posting'] == job]
    model_names = results['model_name'].unique().tolist()

    model_0 = df[df['model_name'] == model_names[0]]
    model_0 = model_0.sort_values(by='score_baseline', ascending=False).head(10)
    model_1 = df[df['model_name'] == model_names[1]]
    model_1 = model_1.sort_values(by='score_baseline', ascending=False).head(10)
    model_2 = df[df['model_name'] == model_names[2]]
    model_2 = model_2.sort_values(by='score_baseline', ascending=False).head(10)

    r0 = set(model_0['resume_A'])
    r1 = set(model_1['resume_A'])
    r2 = set(model_2['resume_A'])

    st.header('Top Candidate Overlap')

    labels = ('Model 0', 'Model 1', 'Model 2')
    colors = ('mediumslateblue', 'hotpink', 'lime')

    fig, ax = plt.subplots()
    venn = venn3([r0, r1, r2], set_labels=labels, set_colors=colors)

    for label in venn.set_labels:
        label.set_color('white')

    st.pyplot(fig, transparent=True)

    st.markdown('***')
    st.write(f'{labels[0]} -- {model_names[0]}')
    st.write(f'{labels[1]} -- {model_names[1]}')
    st.write(f'{labels[2]} -- {model_names[2]}')

def top_candidate_shift(results, job, model):
    df = results[results['job_posting'] == job]
    df = df[df['model_name'] == model]

    baseline = df.sort_values(by='score_baseline', ascending=False).head(10)
    variant = df.sort_values(by='score_variant', ascending=False)

    baseline_top = baseline['resume_A'].tolist()
    variant_top = variant['resume_A_prime'].tolist()

    table = pd.DataFrame(columns=['Resume ID', 'Initial Position', 'New Position', 'Change'])

    resume_scores = {}

    for i, resume in enumerate(baseline_top):
        new_pos = variant_top.index(f'{resume}_prime')
        change = (i+1) - (new_pos+1)
        row = {'Resume ID': resume,
                'Initial Position': i+1,
                'New Position': new_pos+1,
                'Change': f'{change:+}'}
        table.loc[len(table)] = row

        score_baseline = df[df['resume_A'] == resume]['score_baseline'].iloc[0]
        score_variant = df[df['resume_A_prime'] == f'{resume}_prime']['score_variant'].iloc[0]
        resume_scores[resume] = (score_baseline, score_variant)

    st.header('Top Candidate Shift')
    st.table(table)

    resume_names = table['Resume ID'].tolist()
    plot = pd.DataFrame(resume_scores, index=['baseline', 'variant']).T

    x = np.arange(10)
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, plot['baseline'], width, label='baseline', color='indigo')
    ax.bar(x + width/2, plot['variant'], width, label='variant', color='crimson')
    ax.set_ylabel('Similarity Score')
    ax.set_xlabel('Resume ID')
    ax.set_xticks(x)
    ax.set_xticklabels(resume_names)
    ax.tick_params(axis='x', labelrotation=45)
    ax.legend(frameon=False)

    for text in fig.findobj(match=plt.Text):
        text.set_color('white')

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    st.pyplot(fig, transparent=True)

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
    
    st.pyplot(fig, transparent=True)

    resume_index = int(resume_option.split('_')[1])
    job_index = int(job_option.split('_')[1])

    get_resume(resumes, resume_index)
    get_resume_prime(resumes, resume_index)
    get_job_posting(job_postings, job_index)

def resume_to_job_scoring_all(results, job_postings, resumes, job, resume):
    model_names = results['model_name'].unique().tolist()

    df = results[results['job_posting'] == job]

    st.header('Resume-to-Job Scoring')

    model_scores = {}

    for model in model_names:
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

        model_scores[model] = (score_baseline, score_variant)

        st.subheader(model)
        st.table(table)

    plot = pd.DataFrame(model_scores, index=['baseline', 'variant']).T

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, plot['baseline'], width, label=resume, color='indigo')
    ax.bar(x + width/2, plot['variant'], width, label=f'{resume}_prime', color='crimson')
    ax.set_ylabel('Similarity Score')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend(frameon=False)

    for text in fig.findobj(match=plt.Text):
        text.set_color('white')

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    st.pyplot(fig, transparent=True)

    resume_index = int(resume_option.split('_')[1])
    job_index = int(job_option.split('_')[1])

    get_resume(resumes, resume_index)
    get_resume_prime(resumes, resume_index)
    get_job_posting(job_postings, job_index)


if __name__ == '__main__':
    resume_path = 'data/resumes/master_resumes.jsonl'
    job_postings_path = 'data/job_postings/training_data.csv'
    resumes, job_postings = read_files(resume_path, job_postings_path)

    results = pd.read_csv('data/results.csv', index_col=0)

    no_change = results['delta'] == 0
    to_drop = results[no_change].index
    results = results.drop(to_drop)

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
            top_job_overlap(results, resume_option)
        elif job_option is None and resume_option is not None and model_option is not None:
            top_job_shift(results, resume_option, model_option)
        elif job_option is not None and resume_option is None and model_option is None:
            top_candidate_overlap(results, job_option)
        elif job_option is not None and resume_option is None and model_option is not None:
            top_candidate_shift(results, job_option, model_option)
        elif job_option is not None and resume_option is not None and model_option is None:
            resume_to_job_scoring_all(results, job_postings, resumes, job_option, resume_option)
        elif job_option is not None and resume_option is not None and model_option is not None:
            resume_to_job_scoring(results, job_postings, resumes, job_option, resume_option, model_option)
