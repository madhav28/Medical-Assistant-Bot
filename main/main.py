import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel, PeftConfig
from torch import bfloat16
from langchain_core.language_models import LLM
from typing import List
from pydantic import PrivateAttr
import torch

# load vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 4096}
)
vector_store = FAISS.load_local('../vector_store/pubmed_vector_index', embeddings, allow_dangerous_deserialization=True)

# path to the fine-tuned PEFT model
peft_model_id = "../qlora_finetuning/best_model"
# load PEFT configuration
config = PeftConfig.from_pretrained(peft_model_id)
# get the base model ID from the config
base_model_id = config.base_model_name_or_path

tokenizer = AutoTokenizer.from_pretrained(peft_model_id,
                                          cache_dir='/mnt/scratch/lellaom/models',
                                          padding_side="right",
                                          add_eos_token=True)
# set pad token to be the same as EOS token
tokenizer.pad_token = tokenizer.eos_token

# configure 4-bit quantization for efficient inference
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # use 4-bit quantization
    bnb_4bit_quant_type='nf4',  # use normalized float 4 quantization
    bnb_4bit_use_double_quant=True,  # use double quantization for better efficiency
    bnb_4bit_compute_dtype=bfloat16  # use bfloat16 for computations
)

# load the base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=quantization_config,
    device_map="auto",  # automatically distribute model across available devices
    cache_dir='/mnt/scratch/lellaom/models',  # cache directory for model files
)
# finetuned model
model = PeftModel.from_pretrained(base_model, peft_model_id, cache_dir='/mnt/scratch/lellaom/models')
model.eval()

# system prompt
prompt = [{"role": "system", "content": "You are a knowledgeable and reliable assistant specializing in answering medical-related questions using context to ensure accuracy and clarity."}]

# query interface
def ask(questions):
    que = []
    ans = []
    # iterate through each question
    for question in questions:
        question_dict = {"role": "user", "content": question}
        new_prompt = prompt.copy()
        new_prompt.append(question_dict)
        inputs = tokenizer.apply_chat_template(new_prompt, tokenize=True, return_dict=True, return_tensors='pt')
        output_ids = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, do_sample=False, max_new_tokens=500)
        # generated text
        response = tokenizer.decode(output_ids[0], skip_special_tokens=False).split('[/INST]')[-1].replace('</s>', '')

        # retrieving documents for references
        documents = vector_store.similarity_search(question, k=200)
        learnmore1 = []
        learnmore2 = []

        # pubmed references
        for doc in documents:
            id = doc.metadata.get('identifier')
            if isinstance(id, int):
                learnmore1.append(f'{len(learnmore1)+1}. https://pubmed.ncbi.nlm.nih.gov/{id}/\n')
            if len(learnmore1) == 3:
                break

        # mle_screening_dataset.csv references
        for doc in documents:
            id = doc.metadata.get('identifier')
            if isinstance(id, str):
                id = int(id.split('_')[-1])+1
                learnmore2.append(f'{len(learnmore2)+1}. Line {id} in https://github.com/madhav28/Medical-Assistant-Bot/blob/main/data/mle_screening_dataset.csv\n')
            if len(learnmore2) == 3:
                break

        learnmore1 = '\nLearn more from PubMed:\n'+''.join(learnmore1)
        learnmore2 = '\n\nLearn more from mle_screening_dataset.csv:\n'+''.join(learnmore2)
        response += learnmore2+learnmore1

        que.append(question)
        ans.append(response)
    
    # saving responses in results folder
    df = {'question': que, 'answer': ans}
    df = pd.DataFrame(df)
    df.to_csv("../results/bot_answers.csv", index=False)

# Example Queries
ask(["What are the causes and treatments for chronic back pain?",
     "How can I improve my cardiovascular health?",
     "What is Down Syndrome?"])