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

# Data preprocessing
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

# Custom DepthwiseSeparableConv2D implementation
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
        # CIFAR-10: 32x32 -> Conv(same) -> 32x32 -> DepthSep(same) -> 32x32 -> MaxPool -> 16x16
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
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    return epoch_loss, accuracy, precision, recall, f1

# Training and evaluation pipeline
def train_and_evaluate_model(model, model_name, train_loader, val_loader, test_loader, device):
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
    
    # Test evaluation
    print(f"\nFinal Test Evaluation for {model_name}:")
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(
        model, test_loader, criterion, device
    )
    
    results = {
        'model_name': model_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_metrics': {
            'loss': test_loss,
            'accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1
        }
    }
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    # Save model checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = f'checkpoints/{model_name}_checkpoint.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': EPOCHS,
        'test_metrics': results['test_metrics']
    }, checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")
    
    return results

# Main execution
def main():
    all_results = {}
    
    # Train and evaluate Candidate Model
    candidate_model = CandidateModel()
    candidate_results = train_and_evaluate_model(
        candidate_model, 'candidate_model', 
        train_loader, val_loader, test_loader, device
    )
    all_results['candidate_model'] = candidate_results
    
    # Train and evaluate Baseline Model with ReLU
    baseline_model_relu = BaselineModelReLU()
    baseline_results = train_and_evaluate_model(
        baseline_model_relu, 'baseline_model_relu',
        train_loader, val_loader, test_loader, device
    )
    all_results['baseline_model_relu'] = baseline_results
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY: GELU_vs_ReLU_with_DepthwiseConv")
    print(f"{'='*60}")
    print("\nFinal Test Results Comparison:")
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 73)
    
    for model_name, results in all_results.items():
        metrics = results['test_metrics']
        print(f"{model_name:<25} {metrics['accuracy']*100:<11.2f}% {metrics['precision']:<11.4f} "
              f"{metrics['recall']:<11.4f} {metrics['f1_score']:<11.4f}")
    
    # Determine winner
    candidate_acc = all_results['candidate_model']['test_metrics']['accuracy']
    baseline_acc = all_results['baseline_model_relu']['test_metrics']['accuracy']
    
    print(f"\n{'='*60}")
    if candidate_acc > baseline_acc:
        print(f"✓ HYPOTHESIS CONFIRMED: Candidate model with GELU and Depthwise Separable Conv")
        print(f"  outperformed baseline by {(candidate_acc - baseline_acc)*100:.2f}%")
    else:
        print(f"✗ HYPOTHESIS REJECTED: Baseline model with ReLU performed better")
        print(f"  by {(baseline_acc - candidate_acc)*100:.2f}%")
    print(f"{'='*60}")
    
    # Save results to JSON
    os.makedirs('results', exist_ok=True)
    results_path = 'results/experiment_results.json'
    
    # Convert numpy arrays to lists for JSON serialization
    for model_name in all_results:
        all_results[model_name]['train_losses'] = [float(x) for x in all_results[model_name]['train_losses']]
        all_results[model_name]['val_losses'] = [float(x) for x in all_results[model_name]['val_losses']]
        all_results[model_name]['train_accs'] = [float(x) for x in all_results[model_name]['train_accs']]
        all_results[model_name]['val_accs'] = [float(x) for x in all_results[model_name]['val_accs']]
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
```