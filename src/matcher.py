import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from embed_models import get_embedding

def read_files(resume_path, job_postings_path):
    resumes = []
    with open(resume_path, 'r') as file:
        for line in file:
            resumes.append(line.strip())

    df = pd.read_csv(job_postings_path)
    job_postings = df['job_description'].tolist()

    return resumes, job_postings

def match(resume, job_posting, model):
    resume_embedding = get_embedding(resume, model)
    job_posting_embedding = get_embedding(job_posting, model)
    similarity = cosine_similarity(resume_embedding, job_posting_embedding)
    return similarity[0][0]

def match_resumes_to_job(resumes, job_posting, model, n=10):
    similarity_scores = {}
    for i, resume in enumerate(resumes):
        similarity = match(resume, job_posting, model)
        similarity_scores[f'resume_{i}'] = similarity

    top_resumes = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
    top_resumes = top_resumes[0:n]

    return top_resumes

def match_jobs_to_resume(resume, job_postings, model, n=10):
    similarity_scores = {}
    for i, job_posting in enumerate(job_postings):
        similarity = match(resume, job_posting, model)
        similarity_scores[f'job_{i}'] = similarity

    top_jobs = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
    top_jobs = top_jobs[0:n]

    return top_jobs


if __name__ == '__main__':
    resume_path = 'data/resumes/master_resumes.jsonl'
    job_postings_path = 'data/job_postings/training_data.csv'

    resumes, job_postings = read_files(resume_path, job_postings_path)

    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)

    top_resumes = match_resumes_to_job(resumes, job_postings[0], model, 10)
    print(top_resumes)
