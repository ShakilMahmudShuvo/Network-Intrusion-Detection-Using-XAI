"""
Quick test of enhanced preprocessing
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
import json
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Simple model for testing
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

def test_preprocessing(preprocessing_type='enhanced'):
    """Test different preprocessing approaches"""
    print(f"\n=== Testing {preprocessing_type} preprocessing ===\n")
    
    suffix = '_enhanced' if preprocessing_type == 'enhanced' else '_grouped'
    
    try:
        # Load data
        X_train = np.load(f'data/processed/X_train{suffix}.npy')
        X_val = np.load(f'data/processed/X_val{suffix}.npy')
        X_test = np.load(f'data/processed/X_test{suffix}.npy')
        y_train = np.load(f'data/processed/y_train{suffix}.npy')
        y_val = np.load(f'data/processed/y_val{suffix}.npy')
        y_test = np.load(f'data/processed/y_test{suffix}.npy')
        
        with open(f'data/processed/metadata{suffix}.json', 'r') as f:
            metadata = json.load(f)
            
    except FileNotFoundError:
        print(f"Data not found for {preprocessing_type}. Run preprocessing first.")
        return None
    
    print(f"Dataset size: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")
    print(f"Features: {metadata['num_features']}, Classes: {metadata['num_classes']}")
    
    # Create dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    # Create model
    model = SimpleNN(metadata['num_features'], metadata['num_classes']).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    # Quick training
    print("\nTraining...")
    best_val_acc = 0
    
    for epoch in range(30):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        scheduler.step(val_loss / len(val_loader))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Train Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%")
    
    # Test evaluation
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    test_acc = accuracy_score(all_targets, all_preds)
    
    print(f"\nResults for {preprocessing_type}:")
    print(f"Best Val Acc: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    # Per-class results
    print("\nPer-class F1 scores:")
    report = classification_report(all_targets, all_preds, target_names=metadata['classes'], output_dict=True)
    for cls in metadata['classes']:
        if cls in report:
            print(f"  {cls}: {report[cls]['f1-score']:.3f}")
    
    return test_acc

if __name__ == "__main__":
    # Test both preprocessing approaches
    results = {}
    
    # Test grouped preprocessing
    acc_grouped = test_preprocessing('grouped')
    if acc_grouped:
        results['grouped'] = acc_grouped
    
    # Test enhanced preprocessing  
    acc_enhanced = test_preprocessing('enhanced')
    if acc_enhanced:
        results['enhanced'] = acc_enhanced
    
    # Compare results
    if len(results) > 1:
        print("\n" + "="*50)
        print("COMPARISON")
        print("="*50)
        for method, acc in results.items():
            print(f"{method}: {acc*100:.2f}%")
        
        if 'enhanced' in results and 'grouped' in results:
            improvement = (results['enhanced'] - results['grouped']) / results['grouped'] * 100
            print(f"\nImprovement: {improvement:+.1f}%") 