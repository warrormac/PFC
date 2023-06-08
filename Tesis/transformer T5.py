import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, AdamW

# Define the dataset class
class TranslationDataset(Dataset):
    def __init__(self, source_texts, target_texts, tokenizer):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, index):
        source_text = self.source_texts[index]
        target_text = self.target_texts[index]
        encoded_inputs = self.tokenizer(source_text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        encoded_targets = self.tokenizer(target_text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        return {
            'input_ids': encoded_inputs.input_ids.squeeze(),
            'attention_mask': encoded_inputs.attention_mask.squeeze(),
            'decoder_input_ids': encoded_targets.input_ids.squeeze(),
            'decoder_attention_mask': encoded_targets.attention_mask.squeeze(),
        }

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Load the training data
source_texts = ['Phaxsi', 'Jach\'a', 'Mallku', 'Jiska']
target_texts = ['Sun', 'Big', 'Condor', 'Moon']

# Create the dataset and data loader
dataset = TranslationDataset(source_texts, target_texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Configure the T5 model
config = T5Config.from_pretrained('t5-base')
config.num_layers = 6
config.hidden_size = 768  # Set the desired hidden size
config.attention_heads = 8

# Instantiate the T5 model for conditional generation
model = T5ForConditionalGeneration(config)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        decoder_attention_mask = batch['decoder_attention_mask'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=decoder_input_ids
        )

        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}')

# Save the trained model
model.save_pretrained('trained_model')
tokenizer.save_pretrained('trained_model')
