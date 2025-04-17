# Medical-Assistant-Bot

## Dataset Preparation
To improve the quality of the responses, I have augmented approximately 100K PubMed abstracts from 2023 and 2024. I used these 100K abstracts along with the mle_screening_dataset.csv to create a vector store. For QLoRA fine-tuning of Mistral-7B-Instruct-v0.3, I used a sample of 200 question-and-answer pairs as my training, validation, and testing datasets. I split the 200 samples into 160 for training, 20 for validation, and 20 for testing.

### Download PubMed Abstracts
**Download scripts for PubMed Abstracts:** [pubmed_download_scripts](./pubmed_download_scripts)

By running the above scripts, a total of ~ 3M abstracts with PMIDs will be downloaded. After downloading the abstracts, I randomly sampled 100K abstracts for this project.

PubMed Abstracts data: https://michiganstate-my.sharepoint.com/:f:/r/personal/lellaom_msu_edu/Documents/Medical-Assistant-Bot/data?csf=1&web=1&e=UpQXix

## QLoRA Finetuning
**Code:** [qlora_finetuning](./qlora_finetuning)

- **Model & Method**  
  - Finetunes `mistralai/Mistral-7B-Instruct-v0.3` using **QLoRA** (4-bit quantization + Low-Rank Adaptation)  
  - Targets query/key projection matrices for efficient parameter updates  
  - Uses `BitsAndBytesConfig` for 4-bit quantization  

- **Data Processing**  
  - Converts medical Q&A pairs into structured chat format with system prompts  
  - Implements **prompt masking** during tokenization to train only on response generation  

- **Training Configuration**  
  - **Hyperparameters**:  
    - 3 epochs, gradient accumulation (effective batch size=4)  
    - Cosine LR decay (initial rate=3e-5), 10% warmup  
    - bfloat16 precision, weight decay (0.001)  
  - **LoRA Settings**:  
    - Rank-64 adapters, applied to attention layers  

- **Optimizations**  
  - Automatic directory management and dataset splitting (80/10/10 train/val/test)  
  - Periodic validation with best-checkpoint retention  
  - Resource-efficient design for constrained environments  

- **Evaluation**  
  - Monitors validation loss throughout training  
  - Rigorous standards for model performance tracking

**Checkpoints:** https://michiganstate-my.sharepoint.com/:f:/r/personal/lellaom_msu_edu/Documents/Medical-Assistant-Bot/qlora_finetuning/checkpoints?csf=1&web=1&e=IkuXOg<br>
**Best Model:** https://michiganstate-my.sharepoint.com/:f:/r/personal/lellaom_msu_edu/Documents/Medical-Assistant-Bot/qlora_finetuning/best_model?csf=1&web=1&e=4aNo6T

## Vector Store
Code: [vector_store](./vector_store/)

Converted texts (PubMed abstracts and Answers) into a FAISS vector store for efficient semantic search. Key steps:

- **Data Loading & Preprocessing**  
  - Combines two datasets:  
    - PubMed abstracts (`PMID` as identifier, `Abstract` as content)  
    - Q&A pairs (converted to documents with `row_<idx>` identifiers)  
  - Drops empty content and logs total document count  

- **Text Chunking**  
  - Splits documents using a science-aware `RecursiveCharacterTextSplitter`:  
    - Chunk size: 600 tokens  
    - Overlap: 150 tokens  
    - Prioritizes splits at paragraphs (`\n`), sentences (`.\s+`), and clauses (`;,\s+`)  

- **Embedding Generation**  
  - Uses `all-MiniLM-L6-v2` sentence transformer (HuggingFace)  
  - Auto-detects GPU/CPU for acceleration  
  - Normalizes embeddings and processes in large batches (4096)  

- **Vector Store Creation**  
  - Indexes chunks in FAISS for optimized similarity search  
  - Saves output to `pubmed_vector_index` for reuse  

pubmed_vector_index: https://michiganstate-my.sharepoint.com/:f:/r/personal/lellaom_msu_edu/Documents/Medical-Assistant-Bot/vector_store/pubmed_vector_index?csf=1&web=1&e=h2I2hS