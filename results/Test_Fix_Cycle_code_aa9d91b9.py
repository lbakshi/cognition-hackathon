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
import random
import os
import json
from datetime import datetime

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 1
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

# Create validation split
val_size = int(VALIDATION_SPLIT * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

# Model definitions
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # Calculate flattened size: 16 channels * 32 * 32 (CIFAR-10 image size)
        self.fc = nn.Linear(16 * 32 * 32, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class BaselineSimple(nn.Module):
    def __init__(self):
        super(BaselineSimple, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3072, 10)  # 3*32*32 = 3072
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Training function
def train_model(model, train_loader, val_loader, epochs, learning_rate, model_name):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"\nTraining {model_name}...")
    
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
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
    
    return model, train_losses, val_losses, val_accuracies

# Evaluation function
def evaluate_model(model, test_loader, model_name):
    model.eval()
    all_predictions = []
    all_labels = []
    
    print(f"\nEvaluating {model_name}...")
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # For multiclass classification, use weighted average
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return results

# Save checkpoint function
def save_checkpoint(model, model_name, results, epoch):
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'epoch': epoch,
        'results': results
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

# Main experiment
def run_experiment():
    print("="*50)
    print("EXPERIMENT: Test_Fix_Cycle")
    print("HYPOTHESIS: Test the fix functionality")
    print("="*50)
    
    # Dictionary to store all results
    all_results = {}
    
    # Define models
    models = {
        'test_model': TestModel(),
        'baseline_simple': BaselineSimple()
    }
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")
        
        # Print model architecture
        print(f"\nModel Architecture:")
        print(model)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Train model
        trained_model, train_losses, val_losses, val_accuracies = train_model(
            model, train_loader, val_loader, EPOCHS, LEARNING_RATE, model_name
        )
        
        # Evaluate model
        results = evaluate_model(trained_model, test_loader, model_name)
        
        # Save checkpoint
        save_checkpoint(trained_model, model_name, results, EPOCHS)
        
        # Store results
        all_results[model_name] = results
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1_score']:.4f}")
    
    # Print comparison
    print("\n" + "="*50)
    print("FINAL RESULTS COMPARISON")
    print("="*50)
    
    print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*68)
    
    for model_name, results in all_results.items():
        print(f"{model_name:<20} {results['accuracy']:<12.4f} {results['precision']:<12.4f} {results['recall']:<12.4f} {results['f1_score']:<12.4f}")
    
    # Save results to JSON
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'experiment_results_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nResults saved to: {results_file}")
    
    # Determine best model
    best_model = max(all_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest Model: {best_model[0]} with accuracy: {best_model[1]['accuracy']:.4f}")
    
    return all_results

if __name__ == "__main__":
    results = run_experiment()
```