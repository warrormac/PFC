"""#
import torch
import pickle
from transformers import T5Tokenizer, T5Model

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5Model.from_pretrained('t5-base')

# Load the source and target texts
source_texts = ['Phaxsi', 'Jach\'a', 'Mallku', 'Jiska']
target_texts = ['Sun', 'Big', 'Condor', 'Moon']

# Tokenize the source and target texts
tokenized_inputs = tokenizer(source_texts, padding=True, truncation=True, return_tensors='pt')
tokenized_targets = tokenizer(target_texts, padding='longest', truncation=True, return_tensors='pt')

# Encode the input and target sequences
input_ids = tokenized_inputs['input_ids']
attention_mask = tokenized_inputs['attention_mask']
target_ids = tokenized_targets['input_ids']

# Pad the attention mask to match the length of the input sequence
target_attention_mask = tokenized_targets['attention_mask']
target_attention_mask = target_attention_mask[:, :input_ids.shape[1]]

# Encode the input and target sequences
encoded_inputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
encoded_targets = model.encoder(input_ids=target_ids, attention_mask=target_attention_mask)


# Save the encoded input tensor
with open('encoded_inputs.pkl', 'wb') as file:
    pickle.dump(encoded_inputs, file)

# Save the encoded target tensor
with open('encoded_targets.pkl', 'wb') as file:
    pickle.dump(encoded_targets, file)


# Print the encoded tensors
print(encoded_inputs)
print(encoded_targets)
"""

import torch
import pickle
from transformers import T5Tokenizer, T5Model

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5Model.from_pretrained('t5-base')

# Load the source and target texts
source_texts = ['Phaxsi', 'Jach\'a', 'Mallku', 'Jiska']
target_texts = ['Sun', 'Big', 'Condor', 'Moon']

# Tokenize the source and target texts
tokenized_inputs = tokenizer(source_texts, padding=True, truncation=True, return_tensors='pt')
tokenized_targets = tokenizer(target_texts, padding='longest', truncation=True, return_tensors='pt')

# Encode the input sequences
input_ids = tokenized_inputs['input_ids']
attention_mask = tokenized_inputs['attention_mask']
encoded_inputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)

# Encode the target sequences using the decoder part of the model
target_ids = tokenized_targets['input_ids']
target_attention_mask = tokenized_targets['attention_mask']
target_attention_mask = target_attention_mask[:, :input_ids.shape[1]]
encoded_targets = model.decoder(input_ids=target_ids, attention_mask=target_attention_mask)

# Save the encoded input tensor
with open('encoded_inputs.pkl', 'wb') as file:
    pickle.dump(encoded_inputs, file)

# Save the encoded target tensor
with open('encoded_targets.pkl', 'wb') as file:
    pickle.dump(encoded_targets, file)

# Print the encoded tensors
print(encoded_inputs)
print(encoded_targets)


with open('C:/Users/Andre/OneDrive/UCSP/Semestre 7 Online/Proyectos/Tesis/encoded_inputs.pkl', 'rb') as f:
    encoded_input = pickle.load(f)
    
print(encoded_input.keys())

