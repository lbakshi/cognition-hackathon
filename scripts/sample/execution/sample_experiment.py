import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np
from sklearn.metrics import f1_score

# Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, activation='relu'):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train_model(model, train_loader, test_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training_params']['learning_rate'])
    
    loss_history = []
    accuracy_history = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        epoch_loss = running_loss / len(train_loader)
        
        loss_history.append(epoch_loss)
        accuracy_history.append(accuracy)
        
        print(f'Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={accuracy:.4f}')
    
    training_time = time.time() - start_time
    
    # Calculate F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Inference time
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for inputs, _ in test_loader:
            _ = model(inputs.to(device))
            break
    inference_time = (time.time() - start_time) / inputs.size(0)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'loss': epoch_loss,
        'training_time': training_time,
        'inference_time': inference_time,
        'loss_history': loss_history,
        'accuracy_history': accuracy_history
    }

# Main execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config['training_params']['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['training_params']['batch_size'], shuffle=False)

# Train custom model
print("Training custom model...")
custom_model = SimpleCNN(activation=config['model_config'].get('activation', 'relu')).to(device)
custom_results = train_model(
    custom_model,
    train_loader,
    test_loader,
    config['training_params']['epochs'],
    device
)

# Train baseline CNN
print("Training baseline CNN...")
baseline_model = SimpleCNN(activation='relu').to(device)
baseline_results = train_model(
    baseline_model,
    train_loader,
    test_loader,
    config['training_params']['epochs'],
    device
)

# Store results
results['custom_model'] = custom_results
results['baseline_cnn'] = baseline_results

print("Experiment completed!")