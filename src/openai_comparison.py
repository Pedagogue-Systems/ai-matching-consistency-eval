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

    resume = resumes[0]
    job = job_postings[0]

    model = 'text-embedding-3-small'

    resume_embedding = get_embedding(resume, model)
    job_embedding = get_embedding(job, model)
    similarity = cosine_similarity(resume_embedding, job_embedding)

    print('resume_0 vs. job_0')
    print(f'  similarity score: {similarity[0][0]}')
