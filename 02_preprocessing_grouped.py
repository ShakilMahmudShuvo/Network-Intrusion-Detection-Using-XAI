"""
Improved Data Preprocessing with Attack Grouping
Groups similar attacks into broader categories for better classification
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import json
from tqdm import tqdm
import gc
from collections import Counter
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing.feature_selection import AdvancedFeatureSelector

print("=== Improved Data Preprocessing with Attack Grouping ===\n")

# Configuration
SAMPLE_SIZE = 3000000  # 3M samples
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Attack grouping - Group similar attacks into broader categories
ATTACK_GROUPS = {
    'Benign': ['Benign'],
    'DoS/DDoS': ['DoS', 'DDoS'],
    'Scanning/Recon': ['scanning', 'Reconnaissance'],
    'Web Attacks': ['xss', 'injection'],
    'Authentication': ['password', 'Brute Force'],
    'Malware': ['Bot', 'Backdoor', 'ransomware', 'Worms', 'Shellcode'],
    'Exploitation': ['Exploits', 'Infilteration', 'Fuzzers'],
    'Other': ['mitm', 'Generic', 'Analysis', 'Theft']
}

# Create reverse mapping
attack_to_group = {}
for group, attacks in ATTACK_GROUPS.items():
    for attack in attacks:
        attack_to_group[attack] = group

# Non-feature columns
NON_FEATURE_COLS = ['Label', 'Attack', 'Dataset']

print("Loading dataset...")
chunk_size = 100000
chunks = []
total_rows = 0

for chunk in tqdm(pd.read_csv('data/raw/NF-UQ-NIDS-v2.csv', chunksize=chunk_size)):
    chunks.append(chunk)
    total_rows += len(chunk)
    if total_rows >= SAMPLE_SIZE:
        break

df = pd.concat(chunks, ignore_index=True)[:SAMPLE_SIZE]
print(f"Loaded {len(df):,} samples\n")

del chunks
gc.collect()

# Map attacks to groups
print("Grouping attacks into categories...")
df['Attack_Group'] = df['Attack'].map(attack_to_group)

# Check group distribution
group_counts = df['Attack_Group'].value_counts()
print("\nAttack Group Distribution:")
for group, count in group_counts.items():
    print(f"  {group}: {count:,} ({count/len(df)*100:.1f}%)")

# Feature selection
all_features = [col for col in df.columns if col not in NON_FEATURE_COLS + ['Attack_Group']]

# Remove obvious non-predictive features
features_to_remove = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT']
initial_features = [col for col in all_features if col not in features_to_remove]

print(f"\nInitial features: {len(initial_features)}")

# Separate features and target
X_initial = df[initial_features].copy()
y = df['Attack_Group'].copy()

# Handle missing values and infinities
print("\nCleaning data...")
X_initial = X_initial.replace([np.inf, -np.inf], np.nan)
X_initial = X_initial.fillna(0)

# Remove extreme outliers before scaling
print("Removing extreme outliers...")
for col in X_initial.columns:
    q1 = X_initial[col].quantile(0.01)
    q99 = X_initial[col].quantile(0.99)
    X_initial[col] = X_initial[col].clip(lower=q1, upper=q99)

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nNumber of classes: {len(label_encoder.classes_)}")
print(f"Classes: {list(label_encoder.classes_)}")

# Advanced Feature Selection
print("\n=== Performing Advanced Feature Selection ===")
feature_selector = AdvancedFeatureSelector(verbose=True)

# Use a subset for feature selection to speed up the process
sample_indices = np.random.choice(len(X_initial), min(50000, len(X_initial)), replace=False)
X_sample = X_initial.iloc[sample_indices].values
y_sample = y_encoded[sample_indices]

# Fit feature selector
feature_selector.fit(X_sample, y_sample, initial_features)

# Select top features (you can adjust the number or use threshold)
# For 99% accuracy, we might want to keep more features
X, selected_features = feature_selector.select_features(X_initial.values, k=30)
features_to_use = selected_features

print(f"\nSelected {len(features_to_use)} features for training")

# Convert X back to DataFrame for easier handling
X = pd.DataFrame(X, columns=features_to_use)

# Save feature selection report
feature_report = feature_selector.get_feature_report()
feature_report.to_csv('results/feature_selection_report_grouped.csv', index=False)
print("Feature selection report saved to results/feature_selection_report_grouped.csv")

# Plot feature importance
try:
    feature_selector.plot_feature_importance(top_n=20, save_path='results/figures/feature_importance_grouped.png')
except Exception as e:
    print(f"Warning: Could not plot feature importance: {e}")

# Balance dataset with combination of over and undersampling
print("\nBalancing dataset...")
# First apply SMOTE to minority classes
target_samples_per_class = 20000  # Target for each class

# Calculate current distribution
current_dist = Counter(y_encoded)
sampling_strategy_smote = {}

for class_idx, count in current_dist.items():
    if count < target_samples_per_class * 0.5:  # If less than half target, oversample
        sampling_strategy_smote[class_idx] = int(target_samples_per_class * 0.5)

# Apply SMOTE if needed
if sampling_strategy_smote:
    print("Applying SMOTE to minority classes...")
    smote = SMOTE(sampling_strategy=sampling_strategy_smote, random_state=RANDOM_STATE, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
else:
    X_resampled, y_resampled = X, y_encoded

# Recalculate distribution after SMOTE
current_dist_after_smote = Counter(y_resampled)
sampling_strategy_rus = {}

for class_idx, count in current_dist_after_smote.items():
    # Only undersample if current count is greater than target
    if count > target_samples_per_class:
        sampling_strategy_rus[class_idx] = target_samples_per_class

# Apply undersampling only if needed and valid
if sampling_strategy_rus:
    print("Applying undersampling to majority classes...")
    # Check if undersampling is valid
    current_dist_after_smote = Counter(y_resampled)
    valid_sampling = True
    for class_idx, target_count in sampling_strategy_rus.items():
        current_count = current_dist_after_smote[class_idx]
        if target_count > current_count:
            print(f"  Warning: Cannot undersample class {class_idx} to {target_count} (current: {current_count})")
            valid_sampling = False
    
    if valid_sampling:
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy_rus, random_state=RANDOM_STATE)
        X_balanced, y_balanced = rus.fit_resample(X_resampled, y_resampled)
    else:
        print("  Skipping undersampling due to invalid targets")
        X_balanced, y_balanced = X_resampled, y_resampled
else:
    X_balanced, y_balanced = X_resampled, y_resampled

print(f"\nBalanced dataset size: {len(X_balanced):,}")

# Print balanced distribution
balanced_counts = Counter(y_balanced)
print("\nBalanced class distribution:")
for cls_idx, count in balanced_counts.items():
    cls_name = label_encoder.inverse_transform([cls_idx])[0]
    print(f"  {cls_name}: {count:,}")

# Split data
print("\nSplitting dataset...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X_balanced, y_balanced, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE,
    stratify=y_balanced
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=VAL_SIZE/(1-TEST_SIZE),
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print(f"Train set: {len(X_train):,} samples")
print(f"Validation set: {len(X_val):,} samples")
print(f"Test set: {len(X_test):,} samples")

# Feature scaling - Use StandardScaler with careful handling
print("\nScaling features...")
scaler = StandardScaler()

# Fit on train data
X_train_scaled = scaler.fit_transform(X_train)

# Apply log transformation to features with large ranges
for i, col in enumerate(features_to_use):
    if X_train[col].max() > 1000:  # Large range features
        # Log transform (add 1 to avoid log(0))
        X_train_scaled[:, i] = np.log1p(np.abs(X_train_scaled[:, i]))

# Re-standardize after log transform
scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train_scaled)

# Transform validation and test sets
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Apply same log transforms
for i, col in enumerate(features_to_use):
    if X_train[col].max() > 1000:
        X_val_scaled[:, i] = np.log1p(np.abs(X_val_scaled[:, i]))
        X_test_scaled[:, i] = np.log1p(np.abs(X_test_scaled[:, i]))

X_val_scaled = scaler_final.transform(X_val_scaled)
X_test_scaled = scaler_final.transform(X_test_scaled)

# Clip any remaining extreme values
X_train_scaled = np.clip(X_train_scaled, -10, 10)
X_val_scaled = np.clip(X_val_scaled, -10, 10)
X_test_scaled = np.clip(X_test_scaled, -10, 10)

# Save scalers
joblib.dump(scaler, 'results/models/scaler_grouped.pkl')
joblib.dump(scaler_final, 'results/models/scaler_final_grouped.pkl')
joblib.dump(label_encoder, 'results/models/label_encoder_grouped.pkl')

# Feature statistics after scaling
print("\nFeature statistics after scaling (train set):")
print(f"Mean: {X_train_scaled.mean():.3f}")
print(f"Std: {X_train_scaled.std():.3f}")
print(f"Min: {X_train_scaled.min():.3f}")
print(f"Max: {X_train_scaled.max():.3f}")

# Save processed data
print("\nSaving processed data...")
np.save('data/processed/X_train_grouped.npy', X_train_scaled)
np.save('data/processed/X_val_grouped.npy', X_val_scaled)
np.save('data/processed/X_test_grouped.npy', X_test_scaled)
np.save('data/processed/y_train_grouped.npy', y_train)
np.save('data/processed/y_val_grouped.npy', y_val)
np.save('data/processed/y_test_grouped.npy', y_test)

# Save metadata
metadata = {
    'features': features_to_use,
    'num_features': len(features_to_use),
    'num_classes': len(label_encoder.classes_),
    'classes': list(label_encoder.classes_),
    'attack_groups': ATTACK_GROUPS,
    'train_size': len(X_train),
    'val_size': len(X_val),
    'test_size': len(X_test),
    'scaler_type': 'StandardScaler with log transform',
    'balancing_method': 'SMOTE + RandomUnderSampler'
}

with open('data/processed/metadata_grouped.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n=== Preprocessing Complete ===")
print("\nAttack Groups:")
for group, attacks in ATTACK_GROUPS.items():
    print(f"  {group}: {', '.join(attacks)}")
    
print("\nSaved files with '_grouped' suffix") 