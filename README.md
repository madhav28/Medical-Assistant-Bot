# Medical-Assistant-Bot

## Dataset Preparation
To improve the quality of the responses, I have augmented approximately 100K PubMed abstracts from 2023 and 2024. I used these 100K abstracts along with the mle_screening_dataset.csv to create a vector store. For QLoRA fine-tuning of Mistral-7B-Instruct-v0.3, I used a sample of 200 question-and-answer pairs as my training, validation, and testing datasets. I split the 200 samples into 160 for training, 20 for validation, and 20 for testing.

### Download PubMed Abstracts
Download scripts for PubMed Abstracts: [pubmed_download_scripts](./pubmed_download_scripts)

By running the above scripts, a total of ~ 3M abstracts with PMIDs will be downloaded. After downloading the abstracts, I randomly sampled 100K abstracts for this project.

PubMed Abstracts data: https://michiganstate-my.sharepoint.com/:f:/r/personal/lellaom_msu_edu/Documents/Medical-Assistant-Bot/data?csf=1&web=1&e=UpQXix

## QLoRA Finetuning
Code: [qlora_finetuning](./qlora_finetuning)

Efficient fine-tuning of the mistralai/Mistral-7B-Instruct-v0.3 model for medical question-answering is achieved through Quantized Low-Rank Adaptation (QLoRA), combining 4-bit quantization via BitsAndBytesConfig with targeted Low-Rank Adaptation to the model's query and key projection matrices. The pipeline transforms raw medical Q&A pairs into structured chat format complete with system prompts, then applies sophisticated tokenization that masks prompt sections to focus training exclusively on response generation. Training employs gradient accumulation (effective batch size=4), cosine learning rate decay (3e-5 initial rate), and bfloat16 precision across three epochs, with periodic validation and automatic retention of the best-performing checkpoint. The implementation features automatic directory management, intelligent dataset splitting (80/10/10 ratio), and resource-conscious optimizations including rank-64 LoRA adapters, weight decay (0.001), and 10% warmup - enabling effective adaptation of large language models even in resource-constrained environments while maintaining rigorous evaluation standards through validation loss monitoring.

Checkpoints: https://michiganstate-my.sharepoint.com/:f:/r/personal/lellaom_msu_edu/Documents/Medical-Assistant-Bot/qlora_finetuning/checkpoints?csf=1&web=1&e=IkuXOg
Best Model: https://michiganstate-my.sharepoint.com/:f:/r/personal/lellaom_msu_edu/Documents/Medical-Assistant-Bot/qlora_finetuning/best_model?csf=1&web=1&e=4aNo6T

