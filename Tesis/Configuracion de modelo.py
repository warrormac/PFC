import torch
import pickle
from transformers import T5Tokenizer, T5Model, T5Config

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

# Pad the target sequence to match the length of the input sequence
target_ids = target_ids[:, :input_ids.shape[1]]

# Configure the T5 model
config = T5Config.from_pretrained('t5-base')
config.num_layers = 6
config.hidden_size = 768  # Set the desired hidden size
config.attention_heads = 8
model = T5Model(config)

# Use the encoded inputs and targets with the configured T5 model
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    decoder_input_ids=target_ids
)

# Save the encoded input tensor
encoded_inputs = {'last_hidden_state': outputs.last_hidden_state}
with open('encoded_inputs.pkl', 'wb') as file:
    pickle.dump(encoded_inputs, file)

# Save the encoded target tensor
encoded_targets = {'last_hidden_state': outputs.last_hidden_state}
with open('encoded_targets.pkl', 'wb') as file:
    pickle.dump(encoded_targets, file)

# Print the outputs
print(outputs)

# Access specific elements from the outputs
last_hidden_state = outputs.last_hidden_state

# Load the encoded input tensor
with open('encoded_inputs.pkl', 'rb') as file:
    encoded_inputs = pickle.load(file)
    input_tensor = encoded_inputs['last_hidden_state']

# Load the encoded target tensor
with open('encoded_targets.pkl', 'rb') as file:
    encoded_targets = pickle.load(file)
    target_tensor = encoded_targets['last_hidden_state']

# Print the shape of the input and target tensors
print("Input tensor shape:", input_tensor.shape)
print("Target tensor shape:", target_tensor.shape)
