"""
Advanced Training Script for NIDS Models
Includes hyperparameter tuning and ensemble methods
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import KFold
from sklearn.ensemble import VotingClassifier
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from models.novel_architectures import get_model
from models.baseline_models import get_baseline_model

# Configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"=== Advanced Training on {DEVICE} ===\n")

def load_data(preprocessing='grouped'):
    """Load preprocessed data"""
    suffix = '_enhanced' if preprocessing == 'enhanced' else '_grouped'
    
    print(f"Loading {preprocessing} preprocessed data...")
    
    try:
        # Load numpy arrays
        X_train = np.load(f'data/processed/X_train{suffix}.npy')
        X_val = np.load(f'data/processed/X_val{suffix}.npy')
        X_test = np.load(f'data/processed/X_test{suffix}.npy')
        y_train = np.load(f'data/processed/y_train{suffix}.npy')
        y_val = np.load(f'data/processed/y_val{suffix}.npy')
        y_test = np.load(f'data/processed/y_test{suffix}.npy')
        
        # Load metadata
        with open(f'data/processed/metadata{suffix}.json', 'r') as f:
            metadata = json.load(f)
            
    except FileNotFoundError:
        print(f"Data not found for {preprocessing} preprocessing. Using grouped data.")
        # Fallback to grouped
        X_train = np.load('data/processed/X_train_grouped.npy')
        X_val = np.load('data/processed/X_val_grouped.npy')
        X_test = np.load('data/processed/X_test_grouped.npy')
        y_train = np.load('data/processed/y_train_grouped.npy')
        y_val = np.load('data/processed/y_val_grouped.npy')
        y_test = np.load('data/processed/y_test_grouped.npy')
        
        with open('data/processed/metadata_grouped.json', 'r') as f:
            metadata = json.load(f)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Number of features: {metadata['num_features']}")
    print(f"Number of classes: {metadata['num_classes']}")
    print(f"Classes: {metadata['classes']}\n")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, metadata

def create_weighted_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=512):
    """Create DataLoaders with weighted sampling for training"""
    # Calculate class weights for loss function
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / (class_counts + 1e-5)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    
    # Create weighted sampler for training
    sample_weights = [class_weights[label].item() for label in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), 
        torch.LongTensor(y_test)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, class_weights

def create_improved_model(model_name, input_dim, num_classes):
    """Create model with improved configurations"""
    model_kwargs = {}
    
    # Baseline models
    if model_name in ['simple_nn', 'simple_cnn', 'deep_resnet']:
        if model_name == 'simple_nn':
            model_kwargs = {
                'hidden_dims': [256, 128, 64],
                'dropout': 0.3
            }
        elif model_name == 'simple_cnn':
            model_kwargs = {'dropout': 0.3}
        elif model_name == 'deep_resnet':
            model_kwargs = {
                'hidden_dim': 256,
                'num_blocks': 4,
                'dropout': 0.3
            }
        return get_baseline_model(model_name, input_dim, num_classes, **model_kwargs)
    
    # Novel models
    else:
        if model_name == 'hybrid_cnn_transformer':
            model_kwargs = {
                'hidden_dim': 128,  # Smaller hidden dim
                'num_heads': 4,     # Fewer heads
                'dropout': 0.5      # More dropout
            }
        elif model_name == 'temporal_conv_net':
            model_kwargs = {
                'num_channels': [32, 64, 128],  # Smaller channels
                'kernel_size': 3,
                'dropout': 0.3
            }
        elif model_name == 'graph_attention_nids':
            model_kwargs = {
                'hidden_dim': 128,
                'num_heads': 2
            }
        return get_model(model_name, input_dim, num_classes, **model_kwargs)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch with gradient clipping"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def train_model(model_name, train_loader, val_loader, metadata, device, class_weights, 
                lr=0.0001, weight_decay=0.0001, best_hyperparams=None):
    """Train a specific model with improvements"""
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print('='*50)
    
    # Create model using improved configuration
    model = create_improved_model(model_name, metadata['num_features'], metadata['num_classes'])
    model = model.to(device)
    
    # Override with best hyperparameters if provided
    if best_hyperparams:
        # Note: Model architecture can't be changed after creation
        # Only optimizer parameters will be updated below
        pass
    
    # Training constants
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    
    # Use provided learning rate or default
    learning_rate = lr
    weight_decay_val = weight_decay
    
    # Override with best hyperparameters if provided
    if best_hyperparams:
        learning_rate = best_hyperparams.get('lr', learning_rate)
        weight_decay_val = best_hyperparams.get('weight_decay', weight_decay_val)
    
    # Loss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay_val)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Early stopping
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Early stopping based on accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Check early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'input_dim': metadata['num_features'],
        'num_classes': metadata['num_classes'],
        'history': history,
        'best_val_acc': best_val_acc
    }, f'results/models/{model_name}_improved.pth')
    
    return model, history

def evaluate_model(model, test_loader, metadata, device):
    """Comprehensive model evaluation"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    class_report = classification_report(
        all_targets, all_preds, 
        target_names=metadata['classes'],
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'class_report': class_report,
        'confusion_matrix': cm.tolist()
    }
    
    return results

def hyperparameter_tuning(model_name, X_train, y_train, X_val, y_val, metadata, n_trials=50):
    """Hyperparameter tuning using Optuna"""
    print(f"\nHyperparameter tuning for {model_name}...")
    
    def objective(trial):
        # Suggest hyperparameters
        params = {
            'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512, 1024]),
            'lr': trial.suggest_loguniform('lr', 1e-5, 1e-2),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-2),
            'dropout': trial.suggest_uniform('dropout', 0.1, 0.5),
        }
        
        if model_name == 'hybrid_cnn_transformer':
            params['hidden_dim'] = trial.suggest_categorical('hidden_dim', [128, 256, 512])
            params['num_heads'] = trial.suggest_categorical('num_heads', [4, 8, 16])
        elif model_name == 'simple_nn':
            params['hidden_dims'] = trial.suggest_categorical('hidden_dims', 
                [[256, 128, 64], [512, 256, 128], [128, 64, 32]])
        
        # Create model
        if model_name in ['simple_nn', 'simple_cnn', 'deep_resnet']:
            model = get_baseline_model(
                model_name, 
                input_dim=metadata['num_features'],
                num_classes=metadata['num_classes'],
                dropout=params['dropout']
            ).to(DEVICE)
        else:
            model_kwargs = {'dropout': params['dropout']}
            if 'hidden_dim' in params:
                model_kwargs['hidden_dim'] = params['hidden_dim']
            if 'num_heads' in params:
                model_kwargs['num_heads'] = params['num_heads']
                
            model = get_model(
                model_name,
                input_dim=metadata['num_features'],
                num_classes=metadata['num_classes'],
                **model_kwargs
            ).to(DEVICE)
        
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        
        # Quick training for hyperparameter search
        num_epochs = 20
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            # Train
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Validate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            val_acc = correct / total
            best_val_acc = max(best_val_acc, val_acc)
        
        return best_val_acc
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"  Best validation accuracy: {study.best_value:.4f}")
    
    return study.best_params

def main(use_hyperparameter_tuning=False, preprocessing='grouped'):
    """Main training function"""
    # Load data
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, metadata = load_data(preprocessing)
    except FileNotFoundError:
        print(f"Error: {preprocessing} data not found. Please run preprocessing first.")
        return
    
    # Create dataloaders with weighted sampling
    train_loader, val_loader, test_loader, class_weights = create_weighted_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Models to train
    models_to_train = [
        'simple_nn',  # Baseline
        'simple_cnn',  # CNN baseline
        'deep_resnet',  # ResNet baseline
        'hybrid_cnn_transformer',  # Advanced
        'temporal_conv_net'  # Advanced
    ]
    
    # Results storage
    all_results = {}
    best_hyperparams = {}
    
    # Train each model
    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"Processing {model_name}")
        print('='*60)
        
        # Hyperparameter tuning if enabled
        if use_hyperparameter_tuning:
            best_params = hyperparameter_tuning(
                model_name, X_train, y_train, X_val, y_val, metadata, n_trials=20
            )
            best_hyperparams[model_name] = best_params
            
            # Update training parameters
            BATCH_SIZE = best_params.get('batch_size', 512)
            LEARNING_RATE = best_params.get('lr', 0.0001)
            WEIGHT_DECAY = best_params.get('weight_decay', 0.0001)
            
            # Recreate dataloaders with best batch size
            train_loader, val_loader, test_loader, class_weights = create_weighted_dataloaders(
                X_train, X_val, X_test, y_train, y_val, y_test, batch_size=BATCH_SIZE
            )
        else:
            # Default parameters
            BATCH_SIZE = 512
            LEARNING_RATE = 0.0001
            WEIGHT_DECAY = 0.0001
        
        # Train
        model, history = train_model(
            model_name, train_loader, val_loader, metadata, DEVICE, class_weights,
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            best_hyperparams=best_hyperparams.get(model_name, {})
        )
        
        # Evaluate
        print(f"\nEvaluating {model_name}...")
        results = evaluate_model(model, test_loader, metadata, DEVICE)
        
        # Store results
        all_results[model_name] = results
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1_score']:.4f}")
        
        # Print per-class results
        print("\n  Per-class F1-scores:")
        for class_name in metadata['classes']:
            f1 = results['class_report'][class_name]['f1-score']
            support = results['class_report'][class_name]['support']
            print(f"    {class_name}: {f1:.3f} (support: {support})")
    
    # Save all results
    with open('results/improved_model_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Compare models
    print("\n" + "="*50)
    print("Improved Model Comparison:")
    print("="*50)
    print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-"*65)
    for model_name, results in all_results.items():
        print(f"{model_name:<25} {results['accuracy']:<10.4f} "
              f"{results['precision']:<10.4f} {results['recall']:<10.4f} "
              f"{results['f1_score']:<10.4f}")
    
    print("\n✓ Improved training complete! Results saved in results/")

if __name__ == "__main__":
    main() 