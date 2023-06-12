import pandas as pd
import torch
from transformers import T5Tokenizer

df = pd.read_csv('C:/Users/Andre/OneDrive/UCSP/Semestre 7 Online/Proyectos/Tesis/Dataset.csv')

df = df[['aymara', 'english']]

df = df.dropna()

df['prefix_input'] = 'aymara: ' + df['aymara']

source_texts = df['prefix_input'].tolist()
target_texts = df['english'].tolist()



with open('C:/Users/Andre/OneDrive/UCSP/Semestre 7 Online/Proyectos/Tesis/source.txt', 'r') as f:
    source_texts = f.readlines()

with open('C:/Users/Andre/OneDrive/UCSP/Semestre 7 Online/Proyectos/Tesis/target.txt', 'r') as f:
    target_texts = f.readlines()
    
    
# Preprocess the data
prefix = 'aymara:'
source_texts = [prefix + text.strip() for text in source_texts]
target_texts = [text.strip() for text in target_texts]

# Tokenize using T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-base')

source_tokens = tokenizer(source_texts, padding=True, truncation=True, return_tensors='pt')
target_tokens = tokenizer(target_texts, padding=True, truncation=True, return_tensors='pt')

input_ids = source_tokens['input_ids']
attention_mask = source_tokens['attention_mask']
target_ids = target_tokens['input_ids']

# Print the tokenized input
print(input_ids)
print(attention_mask)
print(target_ids)

# Save the tokenized inputs and targets
torch.save(input_ids, 'input_ids.pt')
torch.save(attention_mask, 'attention_mask.pt')
torch.save(target_ids, 'target_ids.pt')


#loader
#input_ids = torch.load('input_ids.pt')
#attention_mask = torch.load('attention_mask.pt')
#target_ids = torch.load('target_ids.pt')
