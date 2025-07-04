# Network Intrusion Detection with Deep Learning and XAI

Research project implementing novel deep learning architectures for network intrusion detection with explainable AI (XAI) analysis.

## Key Innovation: Attack Grouping

Improved classification by grouping 21 attack types into 8 logical categories:
- **Benign**: Normal traffic
- **DoS/DDoS**: Denial of service attacks
- **Scanning/Recon**: Network scanning & reconnaissance
- **Web Attacks**: XSS, SQL injection
- **Authentication**: Password attacks, brute force
- **Malware**: Bot, backdoor, ransomware, worms
- **Exploitation**: Exploits, infiltration, fuzzers
- **Other**: MITM, generic attacks

## Pipeline Overview

The pipeline consists of the following steps (see `run.sh`):

1. **Preprocessing** (`02_preprocessing_enhanced.py` or `02_preprocessing_grouped.py`):
   - Cleans, groups, and engineers features from the raw dataset.
2. **Feature Analysis** (`05_analyze_features.py`):
   - Analyzes feature importance and selection.
3. **Preprocessing Test** (`test_enhanced.py`):
   - Validates the preprocessing output.
4. **Model Training** (`03_train_improved.py`):
   - Trains deep learning models on the processed data.
5. **XAI Analysis** (`04_xai_analysis.py`):
   - Runs explainable AI analysis on model predictions.

## Main Scripts

- `run.sh`: Orchestrates the full pipeline.
- `02_preprocessing_enhanced.py`: Enhanced data preprocessing.
- `02_preprocessing_grouped.py`: Grouped data preprocessing.
- `05_analyze_features.py`: Feature analysis.
- `test_enhanced.py`: Preprocessing validation.
- `03_train_improved.py`: Model training.
- `04_xai_analysis.py`: XAI analysis.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the complete pipeline
chmod +x run.sh
./run.sh
```

## How to Push to GitHub

1. Initialize git (if not already):
   ```bash
   git init
   ```
2. Add pipeline scripts:
   ```bash
   git add run.sh 02_preprocessing_enhanced.py 02_preprocessing_grouped.py 05_analyze_features.py test_enhanced.py 03_train_improved.py 04_xai_analysis.py
   ```
3. Commit your changes:
   ```bash
   git commit -m "Initial commit: NIDS pipeline scripts and core workflow files"
   ```
4. Add your remote and push:
   ```bash
   git remote add origin <your-repo-url>
   git push -u origin master
   ```

## Project Structure

```
├── 02_preprocessing_grouped.py  # Data preprocessing with attack grouping
├── 02_preprocessing_enhanced.py # Enhanced data preprocessing
├── 03_train_improved.py         # Model training
├── 04_xai_analysis.py           # XAI analysis
├── 05_analyze_features.py       # Feature analysis
├── test_enhanced.py             # Preprocessing validation
├── run.sh                       # Complete pipeline script
└── requirements.txt             # Python dependencies
```

## Dataset

Using NF-UQ-NIDS-v2 dataset (76M samples, 44 features) 