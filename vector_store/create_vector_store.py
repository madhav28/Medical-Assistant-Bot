import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch

# load identifiers and content
def load_data(file_path1, file_path2):
    df1 = pd.read_csv(file_path1)
    df1 = df1.rename(columns={'PMID': 'identifier',
                              'Abstract': 'content'})
    df2 = pd.read_csv(file_path2)
    df2 = df2.drop(columns=['question'])
    df2['identifier'] = [f'row_{idx}' for idx in df2.index]
    df2 = df2.rename(columns={'answer': 'content'})
    df = pd.concat([df1, df2])
    df = df.dropna(subset=['content'])
    print(f"Loaded {len(df)} documents")
    
    # convert to langchain documents
    docs = [
        Document(
            page_content=row['content'],
            metadata={"identifier": row['identifier']}
        ) for _, row in df.iterrows()
    ]
    return docs

abstracts = load_data("../data/pubmed_abstracts_2023_and_2024.csv", "../data/mle_screening_dataset.csv")

# text chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=150,
    separators=["\n", "\.\s+", ";\s+", ",\s+", " "],  # science-aware splitting
    length_function=len
)
chunks = text_splitter.split_documents(abstracts)
print(f"Created {len(chunks)} text chunks")

# vector embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 4096}
)
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("pubmed_vector_index")