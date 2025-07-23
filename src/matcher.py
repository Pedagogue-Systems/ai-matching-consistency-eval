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

def match(resume, job_posting, model_name):
    model = SentenceTransformer(model_name)
    resume_embedding = get_embedding(resume, model)
    job_posting_embedding = get_embedding(job_posting, model)
    similarity = cosine_similarity(resume_embedding, job_posting_embedding)
    return similarity[0][0]

def match_resumes_to_job(resumes, job_posting, model_name):
    similarity_scores = []
    for resume in resumes:
        similarity = match(resume, job_posting, model_name)
        similarity_scores.append(similarity)

    return similarity_scores

def match_jobs_to_resume(resume, job_postings, model_name):
    similarity_scores = []
    for job_posting in job_postings:
        similarity = match(resume, job_posting, model_name)
        similarity_scores.append(similarity)

    return similarity_scores


if __name__ == '__main__':
    resume_path = 'data/resumes/master_resumes.jsonl'
    job_postings_path = 'data/job_postings/training_data.csv'

    resumes, job_postings = read_files(resume_path, job_postings_path)

    model_name = 'all-MiniLM-L6-v2'

    # similarity_scores = match_jobs_to_resume(resumes[0], job_postings[0:10], model_name)
    # print(similarity_scores)

    similarity_scores = match_resumes_to_job(resumes[0:10], job_postings[0], model_name)
    print(similarity_scores)
