import torch
import pickle
from transformers import T5Tokenizer, T5Model, T5Config

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
config = T5Config.from_pretrained('t5-base')
model = T5Model(config)

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

# Pad the target sequence to match the length of the input sequence
target_ids = target_ids[:, :input_ids.shape[1]]

# Use the encoded inputs and targets with the configured T5 model
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    decoder_input_ids=target_ids
)

# Save the encoded input tensor
encoded_inputs = outputs.last_hidden_state
torch.save(encoded_inputs, 'encoded_inputs.pt')

# Save the encoded target tensor
encoded_targets = outputs.last_hidden_state
torch.save(encoded_targets, 'encoded_targets.pt')

# Print the outputs
print(outputs)

# Load the encoded input tensor
encoded_inputs = torch.load('encoded_inputs.pt')

# Load the encoded target tensor
encoded_targets = torch.load('encoded_targets.pt')

# Print the shape of the input and target tensors
print("Input tensor shape:", encoded_inputs.shape)
print("Target tensor shape:", encoded_targets.shape)
