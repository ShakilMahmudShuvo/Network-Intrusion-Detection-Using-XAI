"""
Enhanced Data Preprocessing for NIDS
Implements advanced preprocessing techniques for better accuracy
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import OneSidedSelection, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
import joblib
import json
from tqdm import tqdm
import gc
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing.feature_selection import AdvancedFeatureSelector

print("=== Enhanced Data Preprocessing for NIDS ===\n")

# Configuration
SAMPLE_SIZE = 5000000  # Use more data
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Enhanced attack grouping with more careful categorization
ATTACK_GROUPS = {
    'Benign': ['Benign'],
    'DoS': ['DoS', 'DDoS'],  # Keep DoS separate for better precision
    'Probe': ['scanning', 'Reconnaissance'],  # Network probing
    'Web': ['xss', 'injection'],  # Web-based attacks
    'Brute Force': ['password', 'Brute Force'],  # Authentication attacks
    'Malware': ['Bot', 'Backdoor', 'ransomware', 'Worms', 'Shellcode'],
    'Intrusion': ['Exploits', 'Infilteration', 'Fuzzers'],  # Active intrusion
    'Other': ['mitm', 'Generic', 'Analysis', 'Theft']
}

# Create reverse mapping
attack_to_group = {}
for group, attacks in ATTACK_GROUPS.items():
    for attack in attacks:
        attack_to_group[attack] = group

# Enhanced feature handling
FEATURES_TO_DROP = [
    'IPV4_SRC_ADDR',
    'IPV4_DST_ADDR',
    'Label',
    'Attack',
    'Dataset'
]

# Features known to be important for network intrusion detection
IMPORTANT_FEATURES = [
    'IN_BYTES', 'IN_PKTS', 'PROTOCOL', 'TCP_FLAGS',
    'FLOW_DURATION_MILLISECONDS', 'OUT_BYTES', 'OUT_PKTS',
    'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN', 'SRC_TO_DST_AVG_THROUGHPUT'
]

def load_and_clean_data():
    """Load data with better cleaning"""
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
    print(f"Loaded {len(df):,} samples")
    
    del chunks
    gc.collect()
    
    # Map attacks to groups
    print("\nGrouping attacks...")
    df['Attack_Group'] = df['Attack'].map(attack_to_group)
    
    # Remove any unmapped attacks
    df = df.dropna(subset=['Attack_Group'])
    
    # Check distribution
    print("\nAttack Group Distribution:")
    group_counts = df['Attack_Group'].value_counts()
    for group, count in group_counts.items():
        print(f"  {group}: {count:,} ({count/len(df)*100:.1f}%)")
    
    return df

def handle_missing_and_infinite_values(X):
    """Advanced handling of missing and infinite values"""
    print("\nHandling missing and infinite values...")
    
    # Replace infinities with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # For each column, use different strategies
    for col in X.columns:
        if X[col].isna().sum() > 0:
            # If mostly zeros, fill with zero
            if (X[col] == 0).sum() / len(X[col]) > 0.5:
                X[col].fillna(0, inplace=True)
            # For numeric features, use median
            elif X[col].dtype in ['int64', 'float64']:
                X[col].fillna(X[col].median(), inplace=True)
            else:
                X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0, inplace=True)
    
    return X

def remove_outliers_iqr(X, y, contamination=0.1):
    """Remove outliers using IQR method"""
    print("\nRemoving outliers...")
    
    # Calculate IQR for each feature
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Find outliers
    outlier_mask = ((X < lower_bound) | (X > upper_bound)).any(axis=1)
    outlier_percentage = outlier_mask.sum() / len(X) * 100
    
    print(f"  Found {outlier_mask.sum():,} outliers ({outlier_percentage:.1f}%)")
    
    # Remove only if not too many
    if outlier_percentage < contamination * 100:
        X_clean = X[~outlier_mask]
        y_clean = y[~outlier_mask]
        print(f"  Removed outliers. New size: {len(X_clean):,}")
        return X_clean, y_clean
    else:
        print(f"  Too many outliers. Applying clipping instead.")
        for col in X.columns:
            X[col] = X[col].clip(lower=lower_bound[col], upper=upper_bound[col])
        return X, y

def feature_engineering(X):
    """Create new features based on domain knowledge"""
    print("\nEngineering features...")
    
    # Byte/packet ratios
    X['bytes_per_packet_in'] = X['IN_BYTES'] / (X['IN_PKTS'] + 1)
    X['bytes_per_packet_out'] = X['OUT_BYTES'] / (X['OUT_PKTS'] + 1)
    
    # Flow symmetry features
    X['byte_ratio'] = X['IN_BYTES'] / (X['OUT_BYTES'] + 1)
    X['packet_ratio'] = X['IN_PKTS'] / (X['OUT_PKTS'] + 1)
    
    # Duration features
    X['bytes_per_second'] = (X['IN_BYTES'] + X['OUT_BYTES']) / (X['FLOW_DURATION_MILLISECONDS'] / 1000 + 1)
    X['packets_per_second'] = (X['IN_PKTS'] + X['OUT_PKTS']) / (X['FLOW_DURATION_MILLISECONDS'] / 1000 + 1)
    
    # Packet size features
    if 'MIN_IP_PKT_LEN' in X.columns and 'MAX_IP_PKT_LEN' in X.columns:
        X['packet_size_variation'] = X['MAX_IP_PKT_LEN'] - X['MIN_IP_PKT_LEN']
    
    print(f"  Created {len([col for col in X.columns if col not in IMPORTANT_FEATURES])} new features")
    
    return X

def select_features(X, y, n_features=None):
    """Advanced feature selection using multiple methods"""
    print("\nPerforming Advanced Feature Selection...")
    
    # Initialize advanced feature selector
    feature_selector = AdvancedFeatureSelector(verbose=True)
    
    # Use a sample for feature selection if dataset is large
    if len(X) > 100000:
        print("  Using sample of 100k rows for feature selection...")
        sample_indices = np.random.choice(len(X), 100000, replace=False)
        X_sample = X.iloc[sample_indices].values
        y_sample = y[sample_indices]
    else:
        X_sample = X.values
        y_sample = y
    
    # Fit the feature selector
    feature_selector.fit(X_sample, y_sample, X.columns.tolist())
    
    # Select features based on combined scores
    if n_features is None:
        # Use adaptive selection - keep features above average score
        X_selected, selected_features = feature_selector.select_features(X.values, threshold=None)
    else:
        # Select top n features
        X_selected, selected_features = feature_selector.select_features(X.values, k=n_features)
    
    # Save feature selection report
    feature_report = feature_selector.get_feature_report()
    os.makedirs('results', exist_ok=True)
    feature_report.to_csv('results/feature_selection_report_enhanced.csv', index=False)
    print("  Feature selection report saved to results/feature_selection_report_enhanced.csv")
    
    # Plot feature importance
    try:
        os.makedirs('results/figures', exist_ok=True)
        feature_selector.plot_feature_importance(top_n=25, save_path='results/figures/feature_importance_enhanced.png')
    except Exception as e:
        print(f"  Warning: Could not plot feature importance: {e}")
    
    # Convert back to DataFrame
    X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    print(f"  Selected {len(selected_features)} features from {len(X.columns)}")
    
    return X_selected, selected_features

def advanced_sampling(X, y, strategy='hybrid'):
    """Advanced sampling using hybrid techniques"""
    print("\nApplying advanced sampling...")
    
    # Get class distribution
    class_dist = Counter(y)
    print("  Original distribution:", dict(class_dist))
    
    # Calculate target samples
    avg_samples = int(np.mean(list(class_dist.values())))
    target_samples = max(20000, avg_samples)  # At least 20k per class
    
    if strategy == 'hybrid':
        # Step 1: SMOTE for minority classes
        print("  Step 1: Applying BorderlineSMOTE for minority classes...")
        
        # Identify minority classes
        minority_classes = {cls: count for cls, count in class_dist.items() 
                          if count < target_samples * 0.8}
        
        if minority_classes:
            # Use BorderlineSMOTE for better boundary handling
            smote = BorderlineSMOTE(
                sampling_strategy='auto',
                random_state=RANDOM_STATE,
                k_neighbors=min(5, min(minority_classes.values()) - 1)
            )
            X_resampled, y_resampled = smote.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y
        
        # Step 2: Clean with Tomek Links
        print("  Step 2: Cleaning with Tomek Links...")
        tomek = TomekLinks(sampling_strategy='all')
        X_resampled, y_resampled = tomek.fit_resample(X_resampled, y_resampled)
        
        # Step 3: OneSidedSelection for majority classes
        print("  Step 3: Applying OneSidedSelection...")
        oss = OneSidedSelection(
            random_state=RANDOM_STATE,
            n_neighbors=1,
            n_seeds_S=200
        )
        X_balanced, y_balanced = oss.fit_resample(X_resampled, y_resampled)
        
    elif strategy == 'smoteenn':
        # Alternative: SMOTEENN (SMOTE + Edited Nearest Neighbors)
        print("  Using SMOTEENN...")
        smote_enn = SMOTEENN(random_state=RANDOM_STATE)
        X_balanced, y_balanced = smote_enn.fit_resample(X, y)
    
    else:
        # Fallback to standard SMOTE + Tomek
        print("  Using SMOTETomek...")
        smote_tomek = SMOTETomek(random_state=RANDOM_STATE)
        X_balanced, y_balanced = smote_tomek.fit_resample(X, y)
    
    # Final distribution
    final_dist = Counter(y_balanced)
    print("  Final distribution:", dict(final_dist))
    print(f"  Total samples: {len(X_balanced):,}")
    
    return X_balanced, y_balanced

def advanced_scaling(X_train, X_val, X_test, method='power'):
    """Advanced feature scaling"""
    print("\nScaling features...")
    
    if method == 'power':
        # PowerTransformer for non-normal distributions
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
    elif method == 'robust':
        # RobustScaler for outlier resistance
        scaler = RobustScaler(quantile_range=(5, 95))
    elif method == 'minmax':
        # MinMaxScaler for bounded values
        scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        # Standard scaler as fallback
        scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Additional clipping for safety
    clip_value = 5
    X_train_scaled = np.clip(X_train_scaled, -clip_value, clip_value)
    X_val_scaled = np.clip(X_val_scaled, -clip_value, clip_value)
    X_test_scaled = np.clip(X_test_scaled, -clip_value, clip_value)
    
    print(f"  Used {method} scaling")
    print(f"  Train stats - Mean: {X_train_scaled.mean():.3f}, Std: {X_train_scaled.std():.3f}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def main():
    """Main preprocessing pipeline"""
    # Load data
    df = load_and_clean_data()
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in FEATURES_TO_DROP + ['Attack_Group']]
    X = df[feature_cols].copy()
    y = df['Attack_Group'].copy()
    
    # Handle missing values
    X = handle_missing_and_infinite_values(X)
    
    # Feature engineering
    X = feature_engineering(X)
    
    # Remove outliers
    X, y = remove_outliers_iqr(X, y)
    
    # Encode target
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Feature selection
    X_selected, selected_features = select_features(X, y_encoded)
    
    # Advanced sampling
    X_balanced, y_balanced = advanced_sampling(X_selected, y_encoded, strategy='hybrid')
    
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
    
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # Advanced scaling
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = advanced_scaling(
        X_train, X_val, X_test, method='power'
    )
    
    # Save everything
    print("\nSaving preprocessed data...")
    
    # Create directory if not exists
    import os
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('results/models', exist_ok=True)
    
    # Save data
    np.save('data/processed/X_train_enhanced.npy', X_train_scaled)
    np.save('data/processed/X_val_enhanced.npy', X_val_scaled)
    np.save('data/processed/X_test_enhanced.npy', X_test_scaled)
    np.save('data/processed/y_train_enhanced.npy', y_train)
    np.save('data/processed/y_val_enhanced.npy', y_val)
    np.save('data/processed/y_test_enhanced.npy', y_test)
    
    # Save preprocessing objects
    joblib.dump(scaler, 'results/models/scaler_enhanced.pkl')
    joblib.dump(label_encoder, 'results/models/label_encoder_enhanced.pkl')
    
    # Save metadata
    metadata = {
        'features': selected_features,
        'num_features': len(selected_features),
        'num_classes': len(label_encoder.classes_),
        'classes': list(label_encoder.classes_),
        'attack_groups': ATTACK_GROUPS,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'preprocessing': {
            'outlier_removal': 'IQR method',
            'feature_selection': 'MI + ANOVA + RF combined',
            'sampling': 'BorderlineSMOTE + TomekLinks + OSS',
            'scaling': 'PowerTransformer (yeo-johnson)'
        }
    }
    
    with open('data/processed/metadata_enhanced.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n=== Enhanced Preprocessing Complete ===")
    print(f"Features: {len(selected_features)}")
    print(f"Classes: {len(label_encoder.classes_)}")
    print(f"Total samples: {len(X_train) + len(X_val) + len(X_test):,}")

if __name__ == "__main__":
    main() 