# Medical-Assistant-Bot

## 🚀 System Overview
An AI-powered medical assistant combining:
* Fine-tuned LLM: Mistral-7B-Instruct optimized for medical Q&A
* Evidence Engine: FAISS vector store with 100K+ PubMed abstracts
* Clinical Reference System: Links responses to verified sources ([mle_screening_dataset.csv](./data/mle_screening_dataset.csv))

This AI-powered medical assistant combines two key technologies:
* A fine-tuned language model for generating human-like responses
* A document retrieval system for evidence-based reference linking

## 🛠️ Install Requirements
```
pip install -r requirements.txt
```
## 📋 Implementation Workflow
1. Data Preparation
2. QLoRA Finetuning
3. Vector Store
4. Answer Generation
5. Evaluation Metrics
6. Results
7. Future Work

## 📚 Dataset Preparation
To improve the quality of the responses, I have augmented approximately 100K PubMed abstracts from 2023 and 2024. I used these 100K abstracts along with the mle_screening_dataset.csv to create a vector store. For QLoRA fine-tuning of Mistral-7B-Instruct-v0.3, I used a sample of 200 question-and-answer pairs as my training, validation, and testing datasets. I split the 200 samples into 160 for training, 20 for validation, and 20 for testing.

### Download PubMed Abstracts
**Download scripts for PubMed Abstracts:** [pubmed_download_scripts](./pubmed_download_scripts)

By running the above scripts, a total of ~ 3M abstracts with PMIDs will be downloaded. After downloading the abstracts, I randomly sampled 100K abstracts for this project.

PubMed Abstracts data: https://michiganstate-my.sharepoint.com/:f:/r/personal/lellaom_msu_edu/Documents/Medical-Assistant-Bot/data?csf=1&web=1&e=UpQXix

## 🎛️ QLoRA Finetuning
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

## 🗃️ Vector Store
**Code:** [vector_store](./vector_store/)

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

## 💬 Answer Generation
**Code:** [main](./main)
1. **Question Processing**:
   - Formats input with system prompt establishing medical expertise
   - Optimizes tokenization for model compatibility

2. **Response Creation**:
   - Generates answers using fine-tuned medical knowledge
   - Limits responses to 500 tokens for conciseness

3. **Evidence Linking**:
   - Retrieves most relevant documents
   - Selects 3 verified PubMed articles with direct links
   - Includes 3 dataset references from mle screening data

## 🧪 Evaluation Metrics
**Code:** [evaluation_metrics](./evaluation_metrics)
### 📊 Quantitative Analysis
- **Cosine Similarity Comparison**:
  - QLoRA Finetuned Model: `0.80` mean score
  - Base Model: `0.75` mean score  
  *(6.7% absolute improvement, 15% relative improvement)*
  - Metric: Semantic similarity between generated and ground truth responses using sentence embeddings

### 🧠 Qualitative Assessment
- **Blinded ChatGPT Evaluation**:
  - GPT-4-turbo used as independent judge
  - 100% preference for finetuned model responses
  - Evaluation criteria:
    - **Medical Accuracy**: Factual correctness
    - **Clinical Relevance**: Practical applicability
    - **Explanation Quality**: Clarity and structure
    - **Comprehensiveness**: Depth of coverage

### 🔍 Reference Validation
- **Manual Verification Process**:
  - 100% reference relevance confirmed
  - PubMed citations:
    - Cross-checked with original research papers
    - Verified publication dates and study designs
  - Dataset references:
    - Validated line number accuracy (+/- 2 line tolerance)
    - Confirmed clinical context matches

## ✅ Results
### 🧩 Illustration 1
**Question:** What are the causes and treatments for chronic back pain?<br>
**Bot Answer:**<br>
Chronic back pain is a common condition that can be caused by a variety of factors. Here are some of the most common causes:

1. Muscle or ligament strain: This is often caused by heavy lifting, poor posture, or sudden movements.
2. Herniated disc: This occurs when the soft inner material of a disc in the spine bulges out and presses on a nerve.
3. Osteoarthritis: This is a degenerative joint disease that can affect the spine.
4. Skeletal imbalance: This can cause uneven wear and tear on the spine, leading to chronic pain.
5. Spinal stenosis: This is a narrowing of the spinal canal that can put pressure on the spinal cord and nerves.
6. Fractures: Fractures in the spine can cause chronic pain, especially in older adults.
7. Infections: Infections in the spine can cause chronic pain.
8. Degenerative disc disease: This is a condition in which the discs in the spine break down and lose their cushioning ability.

Treatment for chronic back pain depends on the underlying cause. Here are some common treatments:

1. Pain medication: Over-the-counter pain relievers, such as ibuprofen or acetaminophen, can help relieve pain. Prescription pain medication may be necessary for more severe pain.
2. Physical therapy: Physical therapy can help strengthen the muscles in the back and improve flexibility, which can help reduce pain.
3. Exercise: Regular exercise can help improve back strength and flexibility, which can help reduce pain.
4. Heat and cold therapy: Applying heat or cold to the affected area can help relieve pain.
5. Injections: Injections of corticosteroids or other medications can help reduce inflammation and relieve pain.
6. Surgery: In some cases, surgery may be necessary to correct a structural problem in the spine.
7. Lifestyle changes: Making lifestyle changes, such as improving posture, losing weight, and quitting smoking, can help reduce the risk of chronic back pain.
8. Alternative therapies: Alternative therapies, such as acupuncture, massage, and chiropractic care, may help relieve pain.

Learn more from mle_screening_dataset.csv:
1. Line 9595 in https://github.com/madhav28/Medical-Assistant-Bot/blob/main/data/mle_screening_dataset.csv
2. Line 9597 in https://github.com/madhav28/Medical-Assistant-Bot/blob/main/data/mle_screening_dataset.csv
3. Line 2191 in https://github.com/madhav28/Medical-Assistant-Bot/blob/main/data/mle_screening_dataset.csv

Learn more from PubMed:
1. https://pubmed.ncbi.nlm.nih.gov/36958977/
2. https://pubmed.ncbi.nlm.nih.gov/38942979/
3. https://pubmed.ncbi.nlm.nih.gov/36448422/

### 🧩 Illustration 2
**Question:** How can I improve my cardiovascular health?<br>
**Bot Answer:**<br>
Improving cardiovascular health involves adopting a balanced lifestyle that includes regular exercise, a healthy diet, and avoiding harmful habits. Here are some specific steps you can take:

1. Regular Exercise: Aim for at least 30 minutes of moderate-intensity exercise, such as brisk walking, cycling, or swimming, most days of the week. Incorporate strength training exercises into your routine at least twice a week.

2. Healthy Diet: Focus on a diet rich in fruits, vegetables, lean proteins, and whole grains. Limit saturated fats, trans fats, cholesterol, added sugars, and sodium. Consider incorporating foods that are high in fiber, omega-3 fatty acids, and antioxidants.

3. Maintain a Healthy Weight: Excess weight can put a strain on your heart. Aim for a healthy body mass index (BMI) of 18.5-24.9.

4. Limit Alcohol: Drinking too much alcohol can raise blood pressure and lead to other heart problems. If you choose to drink, do so in moderation.

5. Quit Smoking: Smoking damages your heart and blood vessels, increasing your risk of heart disease and stroke.

6. Manage Stress: Chronic stress can contribute to heart disease. Find healthy ways to manage stress, such as meditation, yoga, or deep breathing exercises.

7. Regular Check-ups: Regularly visit your healthcare provider for check-ups. They can monitor your blood pressure, cholesterol levels, and blood sugar levels, and provide guidance on maintaining a healthy heart.

8. Control Chronic Conditions: If you have conditions like diabetes, high blood pressure, or high cholesterol, managing them effectively can help reduce your risk of heart disease.

9. Take Medication as Prescribed: If your healthcare provider prescribes medication to manage your heart health, take it as directed.

10. Get Enough Sleep: Lack of sleep can contribute to high blood pressure, weight gain, and other risk factors for heart disease. Aim for 7-9 hours of sleep per night.

Learn more from mle_screening_dataset.csv:
1. Line 8451 in https://github.com/madhav28/Medical-Assistant-Bot/blob/main/data/mle_screening_dataset.csv
2. Line 8491 in https://github.com/madhav28/Medical-Assistant-Bot/blob/main/data/mle_screening_dataset.csv
3. Line 748 in https://github.com/madhav28/Medical-Assistant-Bot/blob/main/data/mle_screening_dataset.csv

Learn more from PubMed:
1. https://pubmed.ncbi.nlm.nih.gov/39525383/
2. https://pubmed.ncbi.nlm.nih.gov/37882389/
3. https://pubmed.ncbi.nlm.nih.gov/36103036/

### 🧩 Illustration 3
**Question:** What is Down Syndrome?<br>
**Answer:**<br>
Down Syndrome is a genetic disorder caused when abnormal cell division results in an extra full or partial copy of chromosome 21. This extra genetic material causes the developmental changes and physical features of Down Syndrome.

Individuals with Down Syndrome may have a range of physical and developmental challenges, including cognitive delays, distinctive facial features, and a shortened lifespan compared to the general population. However, it's important to note that each person with Down Syndrome is unique, and many can lead healthy, fulfilling lives with appropriate support and care.

Down Syndrome is the most common chromosomal condition diagnosed in infancy, and it occurs in approximately 1 in every 700 live births worldwide. It can affect anyone, regardless of race, gender, or socioeconomic status.

Early intervention, education, and medical care can greatly improve the quality of life for individuals with Down Syndrome. With proper support, many people with Down Syndrome can live independently, work, and participate in community activities.

Learn more from mle_screening_dataset.csv:
1. Line 14017 in https://github.com/madhav28/Medical-Assistant-Bot/blob/main/data/mle_screening_dataset.csv
2. Line 4457 in https://github.com/madhav28/Medical-Assistant-Bot/blob/main/data/mle_screening_dataset.csv
3. Line 1874 in https://github.com/madhav28/Medical-Assistant-Bot/blob/main/data/mle_screening_dataset.csv

Learn more from PubMed:
1. https://pubmed.ncbi.nlm.nih.gov/39194513/
2. https://pubmed.ncbi.nlm.nih.gov/39457216/
3. https://pubmed.ncbi.nlm.nih.gov/39390071/

## 📝 Future Work:
#### **Model Improvements**  
- Scale fine-tuning dataset   
- Experiment with larger base models (e.g., Mistral-8x7B, Llama 3)  
- Add dynamic few-shot learning for context-aware responses  

#### **Retrieval & Evidence**  
- Integrate multi-source evidence (clinical guidelines, drug databases)  
- Prioritize recent research via PubMed timestamp filtering  
- Hybrid search (semantic + keyword) for rare medical terms  

#### **Evaluation & Safety**  
- Automated fact-checking using MedPaLM/BioBERT  
- Adversarial testing against medical misinformation  
- Bias audits (age/race/gender in responses)  

#### **Deployment**  
- FastAPI endpoint for EHR integration  
- Multimodal support (image-based queries)  
- Scheduled PubMed index updates (weekly/monthly)  

#### **UX & Features**  
- Multi-turn conversation memory  
- Urgency detection (e.g., "seek ER care" flags)  
- Multilingual translation support  

#### **Compliance & Scaling**  
- HIPAA-compliant anonymization for PHI  
- Model quantization for edge/offline use  
- Active learning (log uncertain responses for human review) 