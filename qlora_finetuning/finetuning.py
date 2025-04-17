import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil

# delete a directory if it exists
def check_and_delete(dir):
    if os.path.exists(dir) and os.path.isdir(dir):
        shutil.rmtree(dir)
        print(f"Directory '{dir}' has been deleted.")
    else:
        print(f"Directory '{dir}' does not exist.")

# prepare data in chat format
def data_preparation(df):
    # system prompt
    chat = [
        {"role": "system", "content": "You are a knowledgeable and reliable assistant specializing in answering medical-related questions with accuracy and clarity."},
        ]

    # add question as user and response as assistant
    chat_list = []
    for idx, row in df.iterrows():
        question_dict = {"role": "user", "content": row["question"]}
        response_dict = {"role": "assistant", "content": row["answer"]}
        new_chat = chat.copy()
        new_chat.append(question_dict)
        new_chat.append(response_dict)
        # convert dict to string for training
        chat_list.append(tokenizer.apply_chat_template(new_chat, tokenize=False))
    
    df['chat'] = chat_list
    return df

# tokenize dataset
def tokenize_function(examples):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    max_length = 512  # maximum sequence length

    for full_text in examples["chat"]:
        # split text into prompt and response parts
        split_text = full_text.rsplit("[/INST]", maxsplit=1)
        prompt_part = split_text[0] + "[/INST]"  
        tokenized_full = tokenizer(full_text, truncation=True, padding="max_length", max_length=max_length)

        # tokenize just the prompt part to measure its length
        tokenized_prompt = tokenizer(prompt_part, add_special_tokens=False, truncation=False)
        prompt_length = len(tokenized_prompt["input_ids"])
        
        # create labels, masking the prompt part to focus training on the response
        labels = tokenized_full["input_ids"].copy()
        if prompt_length < max_length:
            labels[:prompt_length] = [-100] * prompt_length  # -100 is ignored in loss calculation
        else:
            pass
        
        # helper function to pad or truncate sequences
        def pad_and_truncate(seq, pad_value, length):
            if len(seq) < length:
                return seq + [pad_value] * (length - len(seq))
            return seq[:length]
        
        # process each component to ensure consistent length
        input_ids = pad_and_truncate(tokenized_full["input_ids"], tokenizer.pad_token_id, max_length)
        attention_mask = pad_and_truncate(tokenized_full["attention_mask"], 0, max_length)
        labels = pad_and_truncate(labels, -100, max_length)
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }

# set up directories for outputs and logs
output_dir = "./checkpoints"
logging_dir = "./logs"
check_and_delete(output_dir)
check_and_delete(logging_dir)

# load tokenizer with specific configurations
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id,
                                          cache_dir='/mnt/scratch/lellaom/models',
                                          padding_side="right",
                                          add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token  # use EOS token for padding

# configure 4-bit quantization for memory efficiency
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_use_double_quant=True,  
    bnb_4bit_quant_type="nf4",  
    bnb_4bit_compute_dtype=torch.bfloat16,  
)

# configure LoRA (Low-Rank Adaptation) for efficient fine-tuning
lora_config = LoraConfig(
    r=64,  # rank of the update matrices
    lora_alpha=16,  # scaling factor
    target_modules=["q_proj", "k_proj",],  # modules to apply LoRA to
    lora_dropout=0.1, 
    bias="none",  
    task_type="CAUSAL_LM", 
)

# load the pre-trained model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",  # automatically distribute across available devices
    cache_dir='/mnt/scratch/lellaom/models',  
)

# prepare model for k-bit training and apply LoRA
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# load and prepare the dataset
df = pd.read_csv("../data/mle_screening_dataset.csv")
df = data_preparation(df)  # convert to chat format
df = df.drop_duplicates(subset=['question'], keep='first')  # remove duplicates
df = df.sample(n=200)  # use a subset for training

# split data into train, validation, and test sets
train_df, test_df = train_test_split(df, test_size=0.2)
validation_df, test_df = train_test_split(test_df, test_size=0.5)

# save the splits
train_df.to_csv("../data/mle_screening_train_dataset.csv", index=False)
validation_df.to_csv("../data/mle_screening_validation_dataset.csv", index=False)
test_df.to_csv("../data/mle_screening_test_dataset.csv", index=False)

# prepare datasets for training by removing original columns
train_df = train_df.drop(columns=['question', 'answer'])
validation_df = validation_df.drop(columns=['question', 'answer'])

# convert to HuggingFace Dataset format
train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)

# tokenize the datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True)

# configure training arguments
training_args = TrainingArguments(
    output_dir=output_dir,  
    per_device_train_batch_size=1,  
    per_device_eval_batch_size=1, 
    gradient_accumulation_steps=4,  # cccumulate gradients before updating
    num_train_epochs=3,  
    eval_strategy="epoch",  # evaluate after each epoch
    save_strategy="epoch",  # save model after each epoch
    logging_dir=logging_dir, 
    learning_rate=3e-5,
    lr_scheduler_type="cosine",  # learning rate scheduler
    bf16=True,  # use bfloat16 precision
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  
    greater_is_better=False,
    warmup_ratio=0.1,  # warmup period
    weight_decay=0.001,  # weight decay for regularization
)

# initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    processing_class=tokenizer,
)

# start training
trainer.train()

# save the best model after training
output_dir = "./best_model"
check_and_delete(output_dir)
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)