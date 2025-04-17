from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import numpy as np
import pandas as pd

# Load MiniLM model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def mean_cosine_similarity(column1, column2):
    embeddings1 = model.encode(column1, convert_to_tensor=True)
    embeddings2 = model.encode(column2, convert_to_tensor=True)
    
    similarities = F.cosine_similarity(embeddings1, embeddings2).cpu().numpy()
    return np.mean(similarities)

df1 = pd.read_csv("../data/mle_screening_train_dataset.csv")
df2 = pd.read_csv("../data/mle_screening_train_dataset.csv")

print(f"Base model mean cosine similarity: {mean_cosine_similarity(df1['answer'], df1['generated_answer']):.4f}")
print(f"Finetuned model mean cosine similarity: {mean_cosine_similarity(df2['answer'], df2['generated_answer']):.4f}")