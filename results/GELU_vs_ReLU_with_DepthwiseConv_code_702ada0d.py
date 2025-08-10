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
import json
import os
from datetime import datetime

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directory for saving results
os.makedirs('experiment_results', exist_ok=True)

# Depthwise Separable Convolution implementation
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv2d, self).__init__()
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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.gelu1 = nn.GELU()
        self.depthwise_sep_conv = DepthwiseSeparableConv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.gelu2 = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        
        # Calculate the size after convolutions and pooling
        # CIFAR-10: 32x32 -> Conv(same) -> 32x32 -> Conv(same) -> 32x32 -> MaxPool -> 16x16
        # After maxpool: 64 channels * 16 * 16 = 16384
        # But we need to account for the actual convolution operations
        # Conv1: 32x32 -> 30x30 (kernel=3, no padding specified in original)
        # DepthwiseSep: 30x30 -> 28x28
        # MaxPool: 28x28 -> 14x14
        # So: 64 * 14 * 14 = 12544
        
        # Actually calculate dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 32, 32)
            dummy_output = self._forward_features(dummy_input)
            self.fc_input_size = dummy_output.view(1, -1).size(1)
            
        self.fc = nn.Linear(self.fc_input_size, 10)
        
    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.gelu1(x)
        x = self.depthwise_sep_conv(x)
        x = self.gelu2(x)
        x = self.maxpool(x)
        return x
        
    def forward(self, x):
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Baseline Model with ReLU and standard convolutions
class BaselineModelReLU(nn.Module):
    def __init__(self):
        super(BaselineModelReLU, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        
        # Calculate the size after convolutions and pooling
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 32, 32)
            dummy_output = self._forward_features(dummy_input)
            self.fc_input_size = dummy_output.view(1, -1).size(1)
            
        self.fc = nn.Linear(self.fc_input_size, 10)
        
    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        return x
        
    def forward(self, x):
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Data loading and preprocessing
def load_data(batch_size=64, validation_split=0.2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load training data
    full_train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Split into train and validation
    train_size = int((1 - validation_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # Load test data
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

# Training function
def train_model(model, train_loader, val_loader, epochs=1, lr=0.001):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
    
    return model, train_losses, val_losses

# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
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
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Main experiment function
def run_experiment():
    print("="*80)
    print("EXPERIMENT: GELU_vs_ReLU_with_DepthwiseConv")
    print("="*80)
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = load_data(batch_size=64, validation_split=0.2)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    results = {}
    
    # Define models to train
    models_config = [
        ('candidate_model', CandidateModel(), 'CNN with GELU and Depthwise Separable Convolutions'),
        ('baseline_model_relu', BaselineModelReLU(), 'Standard CNN with ReLU and standard convolutions')
    ]
    
    # Train and evaluate each model
    for model_name, model, description in models_config:
        print("\n" + "="*80)
        print(f"Training {model_name}: {description}")
        print("="*80)
        
        # Train model
        trained_model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, epochs=1, lr=0.001
        )
        
        # Evaluate on validation set
        print(f"\nEvaluating {model_name} on validation set...")
        val_metrics = evaluate_model(trained_model, val_loader)
        
        # Evaluate on test set
        print(f"Evaluating {model_name} on test set...")
        test_metrics = evaluate_model(trained_model, test_loader)
        
        # Save model checkpoint
        checkpoint_path = f'experiment_results/{model_name}_checkpoint.pth'
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }, checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")
        
        # Store results
        results[model_name] = {
            'description': description,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    # Print final results comparison
    print("\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name} ({model_results['description']}):")
        print("-" * 40)
        print("Validation Metrics:")
        for metric, value in model_results['validation_metrics'].items():
            print(f"  {metric}: {value:.4f}")
        print("\nTest Metrics:")
        for metric, value in model_results['test_metrics'].items():
            print(f"  {metric}: {value:.4f}")
    
    # Save results to JSON
    results_json = {}
    for model_name, model_results in results.items():
        results_json[model_name] = {
            'description': model_results['description'],
            'validation_metrics': model_results['validation_metrics'],
            'test_metrics': model_results['test_metrics']
        }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'experiment_results/results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=4)
    print(f"\nResults saved to {results_file}")
    
    # Determine winner
    print("\n" + "="*80)
    print("EXPERIMENT CONCLUSION")
    print("="*80)
    
    candidate_f1 = results['candidate_model']['test_metrics']['f1_score']
    baseline_f1 = results['baseline_model_relu']['test_metrics']['f1_score']
    
    if candidate_f1 > baseline_f1:
        print(f"✓ HYPOTHESIS CONFIRMED: Candidate model (GELU + DepthwiseSeparable) outperforms baseline")
        print(f"  Candidate F1: {candidate_f1:.4f} vs Baseline F1: {baseline_f1:.4f}")
    else:
        print(f"✗ HYPOTHESIS REJECTED: Baseline model performs better or equal")
        print(f"  Candidate F1: {candidate_f1:.4f} vs Baseline F1: {baseline_f1:.4f}")

if __name__ == "__main__":
    run_experiment()
```