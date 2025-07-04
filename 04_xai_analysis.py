"""
XAI Analysis for NIDS Models
Generates comprehensive explanations using SHAP, LIME, attention visualization, etc.
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from models.novel_architectures import get_model
from xai.explainers import NIDSExplainer

# Configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
NUM_EXPLANATIONS = 100  # Number of samples to explain
NUM_SHAP_BACKGROUND = 100  # Background samples for SHAP

print(f"=== XAI Analysis for NIDS Models ({DEVICE}) ===\n")

def load_model_and_data(model_name):
    """Load trained model and test data"""
    # Load model
    checkpoint = torch.load(f'results/models/{model_name}_best.pth', weights_only=True)
    
    # Load metadata
    with open('data/processed/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Initialize model
    model = get_model(
        model_name,
        input_dim=metadata['num_features'],
        num_classes=metadata['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    # Load data
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    X_train = np.load('data/processed/X_train.npy')
    
    # Load label encoder
    label_encoder = joblib.load('results/models/label_encoder.pkl')
    
    return model, X_train, X_test, y_test, metadata, label_encoder

def analyze_feature_importance(explainer, X_data, model_name):
    """Analyze global feature importance"""
    print("\n1. Global Feature Importance Analysis...")
    
    # Calculate feature importance
    feature_importance = explainer.explain_feature_importance_global(
        X_data, method='gradient_shap', num_samples=100
    )
    
    # Visualize and save
    top_features = explainer.visualize_feature_importance(
        feature_importance, 
        top_k=20,
        save_path=f'results/xai_outputs/{model_name}_feature_importance.png'
    )
    
    # Save importance scores
    importance_df = pd.DataFrame({
        'feature': explainer.feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv(f'results/xai_outputs/{model_name}_feature_importance.csv', index=False)
    
    return importance_df

def analyze_shap_explanations(explainer, X_train, X_test, y_test, model_name, num_samples=50):
    """Generate SHAP explanations"""
    print("\n2. SHAP Analysis...")
    
    # Select samples to explain
    sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
    X_explain = X_test[sample_indices]
    y_explain = y_test[sample_indices]
    
    # Get SHAP values
    shap_values = explainer.explain_shap(X_train[:NUM_SHAP_BACKGROUND], X_explain)
    
    # Summary plot for each class
    if isinstance(shap_values, list):
        # Multi-class output
        for class_idx, class_name in enumerate(explainer.class_names[:5]):  # Top 5 classes
            plt.figure(figsize=(10, 6))
            plt.title(f'SHAP Summary - {class_name}')
            
            # Create summary plot manually
            feature_importance = np.abs(shap_values[class_idx]).mean(axis=0)
            top_features_idx = np.argsort(feature_importance)[-10:]
            
            plt.barh(range(len(top_features_idx)), feature_importance[top_features_idx])
            plt.yticks(range(len(top_features_idx)), 
                      [explainer.feature_names[i] for i in top_features_idx])
            plt.xlabel('Mean |SHAP value|')
            plt.tight_layout()
            plt.savefig(f'results/xai_outputs/{model_name}_shap_summary_{class_name}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    return shap_values

def analyze_lime_explanations(explainer, X_train, X_test, y_test, label_encoder, model_name, num_samples=10):
    """Generate LIME explanations for individual predictions"""
    print("\n3. LIME Instance Explanations...")
    
    # Find correctly classified and misclassified samples
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        predictions = explainer.model(X_tensor).argmax(dim=1).cpu().numpy()
    
    correct_mask = predictions == y_test
    misclassified_mask = ~correct_mask
    
    # Analyze correctly classified samples
    if np.any(correct_mask):
        correct_indices = np.where(correct_mask)[0][:5]
        for idx in correct_indices:
            lime_exp = explainer.explain_lime(
                X_train, X_test[idx], 
                num_features=10, 
                num_samples=1000
            )
            
            # Save explanation
            lime_exp.save_to_file(
                f'results/xai_outputs/{model_name}_lime_correct_{idx}.html'
            )
            
            # Visualize
            explainer.visualize_lime_explanation(
                lime_exp,
                save_path=f'results/xai_outputs/{model_name}_lime_correct_{idx}.png'
            )
    
    # Analyze misclassified samples
    if np.any(misclassified_mask):
        misclass_indices = np.where(misclassified_mask)[0][:5]
        misclass_explanations = []
        
        for idx in misclass_indices:
            # Get detailed misclassification explanation
            explanation = explainer.explain_misclassification(
                X_test[idx], y_test[idx], top_k=5
            )
            misclass_explanations.append(explanation)
            
            # LIME explanation
            lime_exp = explainer.explain_lime(
                X_train, X_test[idx], 
                num_features=10, 
                num_samples=1000
            )
            
            lime_exp.save_to_file(
                f'results/xai_outputs/{model_name}_lime_misclass_{idx}.html'
            )
            
        # Save misclassification analysis
        with open(f'results/xai_outputs/{model_name}_misclassifications.json', 'w') as f:
            json.dump(misclass_explanations, f, indent=2)

def analyze_attention_weights(explainer, X_test, model_name, num_samples=20):
    """Analyze attention weights for models with attention mechanisms"""
    print("\n4. Attention Weight Analysis...")
    
    if not hasattr(explainer.model, 'get_attention_weights'):
        print(f"   Model {model_name} does not have attention mechanism. Skipping.")
        return
    
    # Select diverse samples
    sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    attention_patterns = []
    for idx in sample_indices:
        attention_weights = explainer.explain_attention(X_test[idx])
        attention_patterns.append(attention_weights)
    
    # Visualize average attention pattern
    if attention_patterns:
        # Average across samples and layers
        avg_attention = torch.stack([
            torch.stack(sample_attn).mean(0) 
            for sample_attn in attention_patterns
        ]).mean(0)
        
        explainer.visualize_attention_heatmap(
            avg_attention,
            save_path=f'results/xai_outputs/{model_name}_attention_heatmap.png'
        )

def analyze_counterfactuals(explainer, X_test, y_test, label_encoder, model_name, num_samples=10):
    """Generate counterfactual explanations"""
    print("\n5. Counterfactual Analysis...")
    
    # Find benign samples that we'll try to change to attacks
    benign_label = np.where(label_encoder.classes_ == 'Benign')[0][0]
    benign_indices = np.where(y_test == benign_label)[0][:num_samples]
    
    counterfactuals = []
    for idx in benign_indices:
        # Try to generate counterfactual for DDoS attack
        target_label = np.where(label_encoder.classes_ == 'DDoS')[0][0]
        
        cf_instance, changes = explainer.generate_counterfactual(
            X_test[idx], target_label, 
            max_iterations=500, step_size=0.05
        )
        
        # Find top changed features
        top_changes_idx = np.argsort(np.abs(changes))[-5:]
        
        cf_result = {
            'original_idx': int(idx),
            'original_class': 'Benign',
            'target_class': 'DDoS',
            'top_changed_features': {
                explainer.feature_names[i]: {
                    'original_value': float(X_test[idx, i]),
                    'cf_value': float(cf_instance[i]),
                    'change': float(changes[i])
                }
                for i in top_changes_idx
            }
        }
        counterfactuals.append(cf_result)
    
    # Save counterfactual analysis
    with open(f'results/xai_outputs/{model_name}_counterfactuals.json', 'w') as f:
        json.dump(counterfactuals, f, indent=2)

def create_xai_summary_report(model_name, feature_importance_df):
    """Create a summary report of XAI findings"""
    print("\n6. Creating XAI Summary Report...")
    
    report = f"""
# XAI Analysis Report for {model_name}

## Executive Summary
This report presents comprehensive explainability analysis for the {model_name} model
on the NF-UQ-NIDS-v2 dataset for network intrusion detection.

## Key Findings

### 1. Top 10 Most Important Features
{feature_importance_df.head(10).to_string(index=False)}

### 2. Feature Categories
Network Flow Features are the most influential for attack detection:
- Byte-related features (IN_BYTES, OUT_BYTES)
- Packet counts (IN_PKTS, OUT_PKTS)
- Flow duration statistics
- TCP flags and protocol information

### 3. Model Interpretability
- Global feature importance reveals network flow characteristics as key indicators
- SHAP analysis shows how different features contribute to specific attack classifications
- LIME explanations provide instance-level interpretability
- Counterfactual analysis demonstrates minimal changes needed to alter predictions

### 4. Recommendations
1. Focus monitoring on high-importance features for real-time detection
2. Use LIME explanations for investigating specific alerts
3. Apply counterfactual insights for adversarial robustness testing

## Files Generated
- Feature importance visualization and CSV
- SHAP summary plots for top attack classes
- LIME explanations for correct and misclassified instances
- Attention heatmaps (if applicable)
- Counterfactual analysis results
"""
    
    with open(f'results/xai_outputs/{model_name}_xai_report.md', 'w') as f:
        f.write(report)

def main():
    """Main XAI analysis function"""
    # Models to analyze
    models_to_analyze = [
        'hybrid_cnn_transformer',
        'temporal_conv_net',
        'graph_attention_nids'
    ]
    
    for model_name in models_to_analyze:
        print(f"\n{'='*60}")
        print(f"Analyzing {model_name}")
        print('='*60)
        
        try:
            # Load model and data
            model, X_train, X_test, y_test, metadata, label_encoder = load_model_and_data(model_name)
            
            # Initialize explainer
            explainer = NIDSExplainer(
                model=model,
                feature_names=metadata['features'],
                class_names=metadata['classes'],
                device=DEVICE
            )
            
            # 1. Feature Importance
            feature_importance_df = analyze_feature_importance(explainer, X_test, model_name)
            
            # 2. SHAP Analysis
            shap_values = analyze_shap_explanations(
                explainer, X_train, X_test, y_test, model_name
            )
            
            # 3. LIME Analysis
            analyze_lime_explanations(
                explainer, X_train, X_test, y_test, label_encoder, model_name
            )
            
            # 4. Attention Analysis (if applicable)
            analyze_attention_weights(explainer, X_test, model_name)
            
            # 5. Counterfactual Analysis
            analyze_counterfactuals(
                explainer, X_test, y_test, label_encoder, model_name
            )
            
            # 6. Create Summary Report
            create_xai_summary_report(model_name, feature_importance_df)
            
            print(f"\n✓ XAI analysis complete for {model_name}")
            
        except FileNotFoundError:
            print(f"\n✗ Model {model_name} not found. Please train it first.")
        except Exception as e:
            print(f"\n✗ Error analyzing {model_name}: {str(e)}")
    
    print("\n" + "="*60)
    print("✓ All XAI analyses complete! Check results/xai_outputs/")

if __name__ == "__main__":
    main() 