import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import numpy as np
from sklearn.metrics import f1_score

# ResNet18 baseline model
def create_resnet18(num_classes=10):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Depthwise Separable Convolution CNN
class DepthwiseSeparableCNN(nn.Module):
    def __init__(self, activation='gelu'):
        super(DepthwiseSeparableCNN, self).__init__()
        
        # Depthwise separable conv block
        self.depthwise1 = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3)
        self.pointwise1 = nn.Conv2d(3, 32, kernel_size=1)
        
        self.depthwise2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
        self.pointwise2 = nn.Conv2d(32, 64, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x):
        # First depthwise separable block
        x = self.depthwise1(x)
        x = self.pointwise1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)
        
        # Second depthwise separable block
        x = self.depthwise2(x)
        x = self.pointwise2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool(x)
        
        # Fully connected layers
        x = x.view(-1, 64 * 8 * 8)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Training function with early stopping
def train_model_advanced(model, train_loader, test_loader, epochs, device, patience=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training_params']['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    loss_history = []
    accuracy_history = []
    best_accuracy = 0
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
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
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        epoch_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(test_loader)
        
        loss_history.append(val_loss)
        accuracy_history.append(accuracy)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss={epoch_loss:.4f}, Val Loss={val_loss:.4f}, Accuracy={accuracy:.4f}')
        
        # Early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
    
    training_time = time.time() - start_time
    
    # Calculate F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Measure inference time
    model.eval()
    inference_times = []
    
    with torch.no_grad():
        for _ in range(10):  # Average over 10 batches
            for inputs, _ in test_loader:
                start = time.time()
                _ = model(inputs.to(device))
                inference_times.append((time.time() - start) / inputs.size(0))
                break
    
    avg_inference_time = np.mean(inference_times)
    
    return {
        'accuracy': best_accuracy,
        'f1_score': f1,
        'loss': val_loss,
        'training_time': training_time,
        'inference_time': avg_inference_time,
        'loss_history': loss_history,
        'accuracy_history': accuracy_history
    }

# Main execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data augmentation for better training
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=config['training_params']['batch_size'], shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=config['training_params']['batch_size'], shuffle=False, num_workers=2)

print("=" * 50)
print("Training Depthwise Separable CNN with GELU...")
dw_sep_model = DepthwiseSeparableCNN(activation='gelu').to(device)
dw_sep_results = train_model_advanced(
    dw_sep_model,
    train_loader,
    test_loader,
    config['training_params']['epochs'],
    device
)

print("=" * 50)
print("Training ResNet18 baseline...")
resnet_model = create_resnet18().to(device)
resnet_results = train_model_advanced(
    resnet_model,
    train_loader,
    test_loader,
    config['training_params']['epochs'],
    device
)

# Store results
results['depthwise_separable_cnn'] = dw_sep_results
results['resnet18'] = resnet_results

# Calculate improvement metrics
improvement = dw_sep_results['accuracy'] - resnet_results['accuracy']
speedup = resnet_results['training_time'] / dw_sep_results['training_time']

results['comparison'] = {
    'accuracy_improvement': improvement,
    'training_speedup': speedup,
    'best_model': 'depthwise_separable_cnn' if improvement > 0 else 'resnet18'
}

print("=" * 50)
print(f"Accuracy Improvement: {improvement:.4f}")
print(f"Training Speedup: {speedup:.2f}x")
print(f"Best Model: {results['comparison']['best_model']}")
print("Experiment completed!")