import json, re
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from api_key import API_KEY

def get_embedding(text, model):
    client = OpenAI(api_key=API_KEY)
    response = client.embeddings.create(
        model=model,
        input=text
    )
    embed = response.data[0].embedding
    embed = np.array(embed)
    embed = embed.reshape(1, -1)
    return embed

def alter_resume(resume):
    altered_resume = ''
    data = json.loads(resume)
    if 'personal_info' in data and 'summary' in data['personal_info']:
        summary = data['personal_info']['summary']
        sentences = re.split(r'(?<=[.!?]) +', summary)
        data['personal_info']['summary'] = ' '.join(sentences[1:])
        altered_resume = json.dumps(data)
    return altered_resume

def match(job_postings, resumes, model):
    resume_cache = {}
    existing_scores = set()

    model_results = pd.DataFrame(columns=['job_posting', 'resume_A', 'score_baseline',
                                          'resume_A_prime', 'score_variant',
                                          'delta', 'flag', 'model_name'])

    for i, job in enumerate(job_postings):
        job_embedding = get_embedding(job, model)

        for j, resume in enumerate(resumes):
            altered_resume = alter_resume(resume)

            if resume in resume_cache:
                resume_embedding = resume_cache[resume]
            else:
                resume_embedding = get_embedding(resume, model)
                resume_cache[resume] = resume_embedding

            if altered_resume in resume_cache:
                altered_resume_embedding = resume_cache[altered_resume]
            else:
                altered_resume_embedding = get_embedding(altered_resume, model)
                resume_cache[altered_resume] = altered_resume_embedding

            score_baseline = cosine_similarity(resume_embedding, job_embedding)[0][0]
            score_varaint = cosine_similarity(altered_resume_embedding, job_embedding)[0][0]
            delta = score_baseline - score_varaint

            flag = 0
            if abs(delta) > 0.05 and abs(delta) < 0.15:
                flag = 1
            elif abs(delta) >= 0.15:
                flag = 2

            if score_baseline not in existing_scores:
                row = {'job_posting': f'job_{i}',
                       'resume_A': f'resume_{j}',
                       'score_baseline': score_baseline,
                       'resume_A_prime': f'resume_{j}_prime',
                       'score_variant': score_varaint,
                       'delta': delta,
                       'flag': flag,
                       'model_name': model}
                model_results.loc[len(model_results)] = row
                existing_scores.add(score_baseline)

    return model_results

def match_all(job_postings, resumes, models):
    model_results = []
    for model in models:
        model_results.append(match(job_postings, resumes, model))
    
    results = pd.concat(model_results)
    return results

def read_files(resume_path, job_postings_path):
    resumes = []
    with open(resume_path, 'r') as file:
        for line in file:
            resumes.append(line.strip())

    df = pd.read_csv(job_postings_path)
    job_postings = df['job_description'].tolist()

    return resumes, job_postings


if __name__ == '__main__':
    resume_path = 'data/resumes/master_resumes.jsonl'
    job_postings_path = 'data/job_postings/training_data.csv'

    resumes, job_postings = read_files(resume_path, job_postings_path)

    job_postings = job_postings[:50]
    resumes = resumes[:300]

    models = ['text-embedding-3-small',
              'text-embedding-3-large',
              'text-embedding-ada-002']
    
    results = match_all(job_postings, resumes, models)
    results.to_csv('data/openai_results.csv')
    print(results)
