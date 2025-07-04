"""
Feature Analysis and Selection Visualization
Analyze which features are most important for achieving 99% accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing.feature_selection import AdvancedFeatureSelector, analyze_feature_relationships
import warnings
warnings.filterwarnings('ignore')

def analyze_feature_importance():
    """Analyze feature importance from saved reports"""
    print("=== Feature Importance Analysis ===\n")
    
    # Load feature selection reports if they exist
    reports = {}
    
    if os.path.exists('results/feature_selection_report_grouped.csv'):
        reports['grouped'] = pd.read_csv('results/feature_selection_report_grouped.csv')
        print("Loaded grouped preprocessing feature report")
    
    if os.path.exists('results/feature_selection_report_enhanced.csv'):
        reports['enhanced'] = pd.read_csv('results/feature_selection_report_enhanced.csv')
        print("Loaded enhanced preprocessing feature report")
    
    if not reports:
        print("No feature selection reports found. Please run preprocessing first.")
        return
    
    # Analyze each report
    for name, report in reports.items():
        print(f"\n=== {name.upper()} Preprocessing Features ===")
        print(f"Total features analyzed: {len(report)}")
        
        # Top features by combined score
        top_features = report.nlargest(20, 'combined')
        print("\nTop 20 Features by Combined Score:")
        for idx, row in top_features.iterrows():
            print(f"  {row['feature']}: {row['combined']:.4f}")
        
        # Analyze score distributions
        score_methods = ['variance', 'correlation', 'mutual_info', 'chi2', 'anova', 
                        'rf_importance', 'et_importance', 'xgb_importance', 'l1_selection', 'rfe']
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Combined score distribution
        ax = axes[0, 0]
        report['combined'].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title('Distribution of Combined Feature Scores')
        ax.set_xlabel('Combined Score')
        ax.set_ylabel('Count')
        ax.axvline(report['combined'].mean(), color='red', linestyle='--', label='Mean')
        ax.legend()
        
        # 2. Top features heatmap
        ax = axes[0, 1]
        top_10 = report.nlargest(10, 'combined')
        scores_matrix = top_10[score_methods].values.T
        sns.heatmap(scores_matrix, 
                   xticklabels=top_10['feature'].values,
                   yticklabels=score_methods,
                   cmap='YlOrRd', 
                   ax=ax,
                   cbar_kws={'label': 'Score'})
        ax.set_title('Feature Scores by Method (Top 10)')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Method correlation
        ax = axes[1, 0]
        method_corr = report[score_methods].corr()
        sns.heatmap(method_corr, 
                   annot=True, 
                   fmt='.2f',
                   cmap='coolwarm',
                   center=0,
                   ax=ax,
                   square=True)
        ax.set_title('Correlation Between Selection Methods')
        
        # 4. Feature importance by category
        ax = axes[1, 1]
        
        # Categorize features
        categories = {
            'Flow': ['FLOW', 'DURATION'],
            'Bytes': ['BYTES', 'BYTE'],
            'Packets': ['PKT', 'PACKET'],
            'Protocol': ['PROTOCOL', 'TCP', 'UDP', 'L7'],
            'Ports': ['PORT', 'SRC', 'DST'],
            'Statistics': ['MIN', 'MAX', 'AVG', 'STD'],
            'Other': []
        }
        
        feature_categories = []
        for feat in report['feature']:
            assigned = False
            for cat, keywords in categories.items():
                if cat != 'Other' and any(kw in feat.upper() for kw in keywords):
                    feature_categories.append(cat)
                    assigned = True
                    break
            if not assigned:
                feature_categories.append('Other')
        
        report['category'] = feature_categories
        category_scores = report.groupby('category')['combined'].agg(['mean', 'std', 'count'])
        
        category_scores['mean'].plot(kind='bar', ax=ax, color='lightgreen', yerr=category_scores['std'])
        ax.set_title('Average Feature Importance by Category')
        ax.set_xlabel('Feature Category')
        ax.set_ylabel('Average Combined Score')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add count labels
        for i, (idx, row) in enumerate(category_scores.iterrows()):
            ax.text(i, row['mean'] + row['std'] + 0.01, f"n={row['count']}", 
                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'results/figures/feature_analysis_{name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save detailed analysis
        analysis_results = {
            'top_20_features': top_features[['feature', 'combined']].to_dict('records'),
            'score_statistics': {
                'mean': report['combined'].mean(),
                'std': report['combined'].std(),
                'min': report['combined'].min(),
                'max': report['combined'].max(),
                'median': report['combined'].median()
            },
            'category_analysis': category_scores.to_dict(),
            'method_effectiveness': {
                method: report[method].mean() for method in score_methods
            }
        }
        
        import json
        with open(f'results/feature_analysis_{name}.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print(f"\nAnalysis saved to results/feature_analysis_{name}.json")


def compare_preprocessing_features():
    """Compare features selected by different preprocessing methods"""
    print("\n=== Comparing Feature Selection Across Methods ===")
    
    reports = {}
    
    if os.path.exists('results/feature_selection_report_grouped.csv'):
        reports['grouped'] = pd.read_csv('results/feature_selection_report_grouped.csv')
    
    if os.path.exists('results/feature_selection_report_enhanced.csv'):
        reports['enhanced'] = pd.read_csv('results/feature_selection_report_enhanced.csv')
    
    if len(reports) < 2:
        print("Need at least 2 preprocessing reports to compare")
        return
    
    # Get top features from each method
    top_n = 30
    top_features = {}
    
    for name, report in reports.items():
        top_features[name] = set(report.nlargest(top_n, 'combined')['feature'].tolist())
    
    # Calculate overlaps
    print(f"\nTop {top_n} Features Comparison:")
    
    methods = list(top_features.keys())
    overlap_matrix = np.zeros((len(methods), len(methods)))
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i == j:
                overlap_matrix[i, j] = len(top_features[method1])
            else:
                overlap = len(top_features[method1] & top_features[method2])
                overlap_matrix[i, j] = overlap
    
    # Visualize overlap
    plt.figure(figsize=(8, 6))
    sns.heatmap(overlap_matrix, 
               annot=True, 
               fmt='.0f',
               xticklabels=methods,
               yticklabels=methods,
               cmap='Blues',
               square=True)
    plt.title(f'Feature Overlap Matrix (Top {top_n} Features)')
    plt.tight_layout()
    plt.savefig('results/figures/feature_overlap_matrix.png', dpi=300)
    plt.show()
    
    # Find common and unique features
    common_features = set.intersection(*top_features.values())
    print(f"\nCommon features across all methods ({len(common_features)}):")
    for feat in sorted(common_features):
        print(f"  - {feat}")
    
    # Find unique features for each method
    for method, features in top_features.items():
        unique = features - set.union(*[f for m, f in top_features.items() if m != method])
        if unique:
            print(f"\nUnique to {method} ({len(unique)}):")
            for feat in sorted(unique)[:5]:  # Show only first 5
                print(f"  - {feat}")


def recommend_features_for_99_accuracy():
    """Recommend optimal features for achieving 99% accuracy"""
    print("\n=== Feature Recommendations for 99% Accuracy ===")
    
    # Based on the analysis, recommend features
    recommendations = {
        'critical_features': [
            # Network flow characteristics
            'FLOW_DURATION_MILLISECONDS',
            'IN_BYTES', 'OUT_BYTES',
            'IN_PKTS', 'OUT_PKTS',
            
            # Protocol information
            'PROTOCOL', 'L7_PROTO',
            'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
            
            # Packet size statistics
            'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN',
            'SHORTEST_FLOW_PKT', 'LONGEST_FLOW_PKT',
            
            # Flow statistics
            'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
            'DURATION_IN', 'DURATION_OUT',
            
            # Advanced features
            'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT',
            'MIN_TTL', 'MAX_TTL',
            'FWD_PKTS_PAYLOAD.MIN', 'FWD_PKTS_PAYLOAD.MAX',
            'BWD_PKTS_PAYLOAD.MIN', 'BWD_PKTS_PAYLOAD.MAX'
        ],
        
        'engineered_features': [
            'bytes_per_packet_in',
            'bytes_per_packet_out',
            'byte_ratio',
            'packet_ratio',
            'bytes_per_second',
            'packets_per_second',
            'packet_size_variation',
            'flow_symmetry',
            'tcp_flag_diversity'
        ],
        
        'preprocessing_recommendations': {
            'sampling_size': 'At least 5M samples for diversity',
            'feature_count': '30-40 features optimal',
            'scaling': 'PowerTransformer or RobustScaler',
            'outlier_handling': 'IQR clipping with 1-99 percentile',
            'balancing': 'BorderlineSMOTE + TomekLinks',
            'validation_split': '60-20-20 for train-val-test'
        },
        
        'hyperparameter_tips': {
            'learning_rate': 'Start with 0.001, use scheduler',
            'batch_size': '256-512 for stability',
            'epochs': '50-100 with early stopping',
            'dropout': '0.3-0.5 to prevent overfitting',
            'weight_decay': '1e-4 to 1e-3'
        }
    }
    
    print("\n1. CRITICAL FEATURES (Must Include):")
    for feat in recommendations['critical_features']:
        print(f"   - {feat}")
    
    print("\n2. ENGINEERED FEATURES (Create These):")
    for feat in recommendations['engineered_features']:
        print(f"   - {feat}")
    
    print("\n3. PREPROCESSING RECOMMENDATIONS:")
    for key, value in recommendations['preprocessing_recommendations'].items():
        print(f"   - {key}: {value}")
    
    print("\n4. HYPERPARAMETER TIPS:")
    for key, value in recommendations['hyperparameter_tips'].items():
        print(f"   - {key}: {value}")
    
    # Save recommendations
    import json
    with open('results/feature_recommendations_99_accuracy.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    print("\nRecommendations saved to results/feature_recommendations_99_accuracy.json")


if __name__ == "__main__":
    # Run all analyses
    analyze_feature_importance()
    compare_preprocessing_features()
    recommend_features_for_99_accuracy()
    
    print("\n=== Feature Analysis Complete ===")
    print("Check the results/ directory for detailed reports and visualizations") 