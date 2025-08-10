```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
import json
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
VALIDATION_SPLIT = 0.2

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
print("Loading CIFAR-10 dataset...")
full_train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

# Split training data into train and validation
train_size = int((1 - VALIDATION_SPLIT) * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

# Custom Depthwise Separable Convolution
class DepthwiseSeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv2D, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=True
        )
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Candidate Model with GELU and Depthwise Separable Convolutions
class CandidateModel(nn.Module):
    def __init__(self):
        super(CandidateModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.gelu1 = nn.GELU()
        self.depthwise_sep = DepthwiseSeparableConv2D(32, 64, kernel_size=3, padding=1)
        self.gelu2 = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        # Calculate the size after convolutions and pooling
        # CIFAR-10: 32x32 -> Conv(same) -> 32x32 -> Conv(same) -> 32x32 -> MaxPool -> 16x16
        self.fc = nn.Linear(64 * 16 * 16, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu1(x)
        x = self.depthwise_sep(x)
        x = self.gelu2(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Baseline Model with ReLU and standard convolutions
class BaselineModelReLU(nn.Module):
    def __init__(self):
        super(BaselineModelReLU, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        # Calculate the size after convolutions and pooling
        # CIFAR-10: 32x32 -> Conv(same) -> 32x32 -> Conv(same) -> 32x32 -> MaxPool -> 16x16
        self.fc = nn.Linear(64 * 16 * 16, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Training function
def train_model(model, train_loader, val_loader, model_name, epochs=EPOCHS):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"\nTraining {model_name}...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({'loss': loss.item(), 'acc': 100.*train_correct/train_total})
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * train_correct / train_total
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_bar.set_postfix({'loss': loss.item(), 'acc': 100.*val_correct/val_total})
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
    
    # Save model checkpoint
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
    }, os.path.join(checkpoint_dir, f'{model_name}_checkpoint.pth'))
    
    return model, train_losses, val_losses, val_accuracies

# Evaluation function
def evaluate_model(model, test_loader, model_name):
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    print(f"\nEvaluating {model_name}...")
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics

# Main execution
def main():
    results = {}
    
    # Train and evaluate candidate model
    print("\n" + "="*50)
    print("CANDIDATE MODEL: CNN with GELU and Depthwise Separable Convolutions")
    print("="*50)
    
    candidate_model = CandidateModel()
    trained_candidate, train_losses_c, val_losses_c, val_acc_c = train_model(
        candidate_model, train_loader, val_loader, 'candidate_model'
    )
    candidate_metrics = evaluate_model(trained_candidate, test_loader, 'candidate_model')
    results['candidate_model'] = candidate_metrics
    
    # Train and evaluate baseline model
    print("\n" + "="*50)
    print("BASELINE MODEL: Standard CNN with ReLU")
    print("="*50)
    
    baseline_model = BaselineModelReLU()
    trained_baseline, train_losses_b, val_losses_b, val_acc_b = train_model(
        baseline_model, train_loader, val_loader, 'baseline_model_relu'
    )
    baseline_metrics = evaluate_model(trained_baseline, test_loader, 'baseline_model_relu')
    results['baseline_model_relu'] = baseline_metrics
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    # Compare models
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    candidate_acc = results['candidate_model']['accuracy']
    baseline_acc = results['baseline_model_relu']['accuracy']
    
    if candidate_acc > baseline_acc:
        improvement = ((candidate_acc - baseline_acc) / baseline_acc) * 100
        print(f"✓ Candidate model OUTPERFORMS baseline by {improvement:.2f}%")
    else:
        degradation = ((baseline_acc - candidate_acc) / baseline_acc) * 100
        print(f"✗ Baseline model outperforms candidate by {degradation:.2f}%")
    
    # Save results to JSON
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'experiment_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {os.path.join(results_dir, 'experiment_results.json')}")
    print(f"Model checkpoints saved to checkpoints/")
    
    return results

if __name__ == "__main__":
    results = main()
```