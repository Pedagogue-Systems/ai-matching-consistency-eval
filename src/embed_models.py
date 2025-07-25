import numpy as np

from sentence_transformers import SentenceTransformer

def get_embedding(text, model):
    embedding = model.encode(text)
    embedding = np.array(embedding)
    embedding = embedding.reshape(1, -1)
    return embedding


if __name__ == '__main__':
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = get_embedding('Test embedding.', model)
    print(embedding.shape)
    print(embedding)
