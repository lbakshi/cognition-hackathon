import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import json
from collections import Counter

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the CNN model based on the specification
class CNNTextClassifier(nn.Module):
    def __init__(self):
        super(CNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=10000, embedding_dim=128)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=100, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.adaptive_maxpool = nn.AdaptiveMaxPool1d(output_size=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=100, out_features=50)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(in_features=50, out_features=1)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, sequence_length)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.adaptive_maxpool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# Custom Dataset class for IMDB
class IMDBDataset(Dataset):
    def __init__(self, data, vocab, tokenizer, max_length=500):
        self.data = list(data)
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label, text = self.data[idx]
        # Convert label: 'pos' -> 1, 'neg' -> 0
        label = 1 if label == 'pos' else 0
        
        # Tokenize and convert to indices
        tokens = self.tokenizer(text)
        indices = [self.vocab[token] for token in tokens]
        
        # Truncate or pad
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices = indices + [self.vocab['<pad>']] * (self.max_length - len(indices))
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float)

# Load and prepare data
tokenizer = get_tokenizer('basic_english')

# Load IMDB dataset
train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

# Build vocabulary
def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

# Recreate iterators for vocabulary building
train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>', '<pad>'], max_tokens=10000)
vocab.set_default_index(vocab['<unk>'])

# Recreate iterators for dataset creation
train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

# Create datasets
train_dataset = IMDBDataset(train_iter, vocab, tokenizer)
test_dataset = IMDBDataset(test_iter, vocab, tokenizer)

# Split training data into train and validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss, and optimizer
model = CNNTextClassifier().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data).squeeze()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Store predictions for metrics
        preds = torch.sigmoid(output) > 0.5
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
    
    return total_loss / len(loader), all_preds, all_labels

# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            loss = criterion(output, target)
            
            total_loss += loss.item()
            
            # Store predictions for metrics
            preds = torch.sigmoid(output) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    return total_loss / len(loader), all_preds, all_labels

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }

# Training loop
num_epochs = 10
best_val_acc = 0

for epoch in range(num_epochs):
    # Train
    train_loss, train_preds, train_labels = train_epoch(model, train_loader, criterion, optimizer, device)
    train_metrics = calculate_metrics(train_labels, train_preds)
    
    # Validate
    val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
    val_metrics = calculate_metrics(val_labels, val_preds)
    
    # Update best model
    if val_metrics['accuracy'] > best_val_acc:
        best_val_acc = val_metrics['accuracy']
        best_model_state = model.state_dict()

# Load best model and evaluate on test set
model.load_state_dict(best_model_state)
test_loss, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
test_metrics = calculate_metrics(test_labels, test_preds)

# Print final results as JSON
final_results = {
    "status": "completed",
    "metrics": {
        "accuracy": float(test_metrics['accuracy']),
        "precision": float(test_metrics['precision']),
        "recall": float(test_metrics['recall']),
        "f1_score": float(test_metrics['f1_score'])
    }
}

print(json.dumps(final_results))
