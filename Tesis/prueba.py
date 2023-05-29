import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Transformer

# Set the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tokenizers for Aymara and English languages
def tokenizer(text):
    return text.strip().split()

# Load the dataset
dataset_path = 'C:/Users/Andre/OneDrive/UCSP/Semestre 7 Online/Proyectos/Tesis/Dataset.csv'
aymara_sentences = []
english_sentences = []

with open(dataset_path, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        aymara_sentences.append(row['aymara'])
        english_sentences.append(row['english'])

# Tokenize Aymara sentences
aymara_tokenized = [tokenizer(sentence) for sentence in aymara_sentences]

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransformerModel, self).__init__()
        self.transformer = Transformer(...)
        self.fc = nn.Linear(...)

    def forward(self, src, tgt):
        # Implement the forward pass of the model
        ...

# Initialize the model
model = TransformerModel(len(aymara_tokenized), len(english_sentences)).to(device)

# Set the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Modify the ignore_index value if needed
optimizer = optim.Adam(model.parameters())

# Split the dataset into train, validation, and test sets
split_ratio = [0.7, 0.2, 0.1]
train_data = aymara_tokenized[:int(split_ratio[0] * len(aymara_tokenized))]
valid_data = aymara_tokenized[int(split_ratio[0] * len(aymara_tokenized)):int((split_ratio[0] + split_ratio[1]) * len(aymara_tokenized))]
test_data = aymara_tokenized[int((split_ratio[0] + split_ratio[1]) * len(aymara_tokenized)):]

# Create iterators for the train, validation, and test sets
def create_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

batch_size = 64
train_iterator = create_batches(train_data, batch_size)
valid_iterator = create_batches(valid_data, batch_size)
test_iterator = create_batches(test_data, batch_size)

# Training loop
num_epochs = 10  # Set the number of epochs
for epoch in range(num_epochs):
    model.train()
    for batch in train_iterator:
        src = batch
        tgt = batch  # Modify this line with the correct target data

        optimizer.zero_grad()
        output = model(src, tgt[:-1])
        output = output.view(-1, output.shape[-1])
        tgt = tgt[1:].view(-1)

        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
