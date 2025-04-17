import os
import pandas as pd
from torch import bfloat16
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import joblib
from huggingface_hub import login
import torch
import re
from tqdm import tqdm
from peft import PeftModel, PeftConfig
torch.cuda.empty_cache()


# Function to prepare data in chat format for the model
def data_preparation(df):
    # System message defining the assistant's role
    chat = [
        {"role": "system", "content": "You are a knowledgeable and reliable assistant specializing in answering medical-related questions with accuracy and clarity."},
        ]

    # Convert each Q&A pair into proper chat format
    chat_list = []
    for idx, row in df.iterrows():
        question_dict = {"role": "user", "content": row["question"]}
        new_chat = chat.copy()
        new_chat.append(question_dict)
        chat_list.append(new_chat)
    
    df['chat'] = chat_list
    return df

# Path to the fine-tuned PEFT model
peft_model_id = "../qlora_finetuning/best_model"
# Load PEFT configuration
config = PeftConfig.from_pretrained(peft_model_id)
# Get the base model ID from the config
base_model_id = config.base_model_name_or_path

# Initialize tokenizer with specific settings
tokenizer = AutoTokenizer.from_pretrained(peft_model_id,
                                          cache_dir='/mnt/scratch/lellaom/models',
                                          padding_side="right",
                                          add_eos_token=True)
# Set pad token to be the same as EOS token
tokenizer.pad_token = tokenizer.eos_token

# Configure 4-bit quantization for efficient inference
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Use 4-bit quantization
    bnb_4bit_quant_type='nf4',  # Use normalized float 4 quantization
    bnb_4bit_use_double_quant=True,  # Use double quantization for better efficiency
    bnb_4bit_compute_dtype=bfloat16  # Use bfloat16 for computations
)

# Load the base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=quantization_config,
    device_map="auto",  # Automatically distribute model across available devices
    cache_dir='/mnt/scratch/lellaom/models',  # Cache directory for model files
)

model = PeftModel.from_pretrained(base_model, peft_model_id, cache_dir='/mnt/scratch/lellaom/models')

# Set model to evaluation mode
model.eval()

# Load test dataset
df = pd.read_csv("../data/mle_screening_test_dataset.csv")
# Prepare data in chat format
df = data_preparation(df)

# Generate answers for each question
gen_ans = []
for chat in tqdm(df['chat'], desc="Inference Running"):
    # Tokenize the chat and prepare for model input
    inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, 
                                           return_dict=True, return_tensors='pt').to(model.device)
    # Generate output from the model
    output_ids = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, do_sample=False, max_new_tokens=250)
    # Decode and extract the generated answer
    gen_ans.append(tokenizer.decode(output_ids[0], skip_special_tokens=False).split('[/INST]')[-1])    

df = df.drop(columns=['chat'])
df['generated_answer'] = gen_ans

df.to_csv("../data/mle_screening_test_dataset_finetuned_prediction.csv", index=False)