import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Define your custom dataset
class CustomDataset(Dataset):
    def __init__(self, encoded_inputs_file, encoded_targets_file):
        self.encoded_inputs = torch.load(encoded_inputs_file)
        self.encoded_targets = torch.load(encoded_targets_file)

    def __len__(self):
        return len(self.encoded_inputs)

    def __getitem__(self, index):
        input_data = self.encoded_inputs[index]
        target_data = self.encoded_targets[index]
        return input_data, target_data

# Define your custom transformer model
class CustomTransformer(nn.Module):
    def __init__(self, input_dim, target_dim, hidden_dim, num_layers):
        super(CustomTransformer, self).__init__()

        self.encoder = nn.TransformerEncoderLayer(input_dim, nhead=8, dim_feedforward=hidden_dim)
        self.decoder = nn.TransformerDecoderLayer(target_dim, nhead=8, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(self.encoder, num_layers=num_layers)

    def forward(self, inputs, targets):
        encoded_inputs = self.transformer(inputs)
        decoded_outputs = self.decoder(targets, encoded_inputs)
        return decoded_outputs

# Define hyperparameters
input_dim = 768  # Specify the input dimension based on the shape of your encoded inputs
target_dim = 768  # Specify the target dimension based on the shape of your encoded targets
hidden_dim = 512
num_layers = 4
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print('GPU is available!')
else:
    print('Using CPU for training.')

# Create dataloader for training
dataset = CustomDataset('C:/Users/Andre/OneDrive/UCSP/Semestre 7 Online/Proyectos/Tesis/encoded_inputs.pt', 'C:/Users/Andre/OneDrive/UCSP/Semestre 7 Online/Proyectos/Tesis/encoded_targets.pt')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create the model and optimizer
model = CustomTransformer(input_dim, target_dim, hidden_dim, num_layers)
model.to(device)  # Move the model to GPU if available
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)  # Move inputs to GPU if available
        targets = targets.to(device)  # Move targets to GPU if available

        optimizer.zero_grad()
        outputs = model(inputs, targets)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pt')



print(torch.cuda.is_available())
print(torch.cuda.device_count())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
inputs = inputs.to(device)
targets = targets.to(device)
