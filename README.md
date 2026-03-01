# GRU-Based Real-Time Web Threat Detection

A deep learning model using **Bidirectional Gated Recurrent Units (GRUs)** to detect malicious web requests in real-time. The model treats request payloads as character-level sequences and learns to identify patterns indicative of SQL Injection (SQLi), Cross-Site Scripting (XSS), Command Injection, Path Traversal, SSTI, and other web attacks.

Trained on a large-scale dataset aggregated from six sources (five public datasets plus augmented samples) with comprehensive regularization to prevent overfitting.

## 🎯 Key Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **97.86%** |
| Precision (Benign) | 98% |
| Precision (Malicious) | 98% |
| Recall (Benign) | 99% |
| Recall (Malicious) | 96% |
| F1 (macro avg) | 98% |
| Train/Val Loss Gap | 0.006 (no overfitting) |

## ✨ Features

- **Bidirectional Character-Level GRU Network**: Learns sequential patterns in both directions directly from raw payloads without manual feature engineering
- **Large-Scale Training**: Samples from 6 diverse sources (XSS, SQLi, CSIC 2010, malicious URLs, master web attacks, augmented data)
- **High Performance**: 97.86% accuracy on a held-out test set with balanced precision/recall
- **Overfitting Prevention**: L2 regularization, spatial & dense dropout, learning rate scheduling, early stopping, class weighting, and proper train/val/test split
- **Apple Silicon Optimized**: Configured for Metal GPU acceleration on M-series chips
- **Interactive Testing**: CLI tool with single, batch, demo, and interactive prediction modes
- **Full Metrics Suite**: Confusion matrices, ROC curve, Precision-Recall curve, training curves, learning rate schedule, and per-class metrics

## 🧠 Model Architecture

Sequential Keras model for binary classification (benign vs. malicious):

```
Input (char sequences, max_len=300)
  → Embedding(vocab_size, 128)
  → SpatialDropout1D(0.15)
  → Bidirectional(GRU(128, return_sequences=True, dropout=0.2, L2=1e-5))
  → GRU(128, dropout=0.2, L2=1e-5)
  → Dense(64, ReLU, L2=1e-5)
  → Dropout(0.3)
  → Dense(32, ReLU, L2=1e-5)
  → Dropout(0.2)
  → Dense(1, Sigmoid)
```

**Key Design Choices:**
1. **Character-Level Tokenization**: Captures character patterns in attack payloads (e.g., `<script>`, `' OR 1=1`, `{{7*7}}`)
2. **Bidirectional GRU**: First GRU layer reads sequences in both directions for richer context
3. **Stacked GRU Layers**: Second GRU captures higher-level sequential patterns
4. **SpatialDropout1D (15%)**: Drops entire embedding dimensions to prevent co-adaptation
5. **L2 Regularization (1e-5)**: Light regularization applied to GRU kernels and Dense layers
6. **ReduceLROnPlateau**: Halves learning rate after 3 epochs of val_loss stagnation
7. **EarlyStopping (patience=7)**: Restores best weights to prevent overfitting
8. **Class Weighting**: Computed automatically to handle label imbalance

## 📊 Dataset

Combined from six sources, cleaned and deduplicated:

| Dataset | Description |
|---------|-------------|
| XSS_dataset.csv | Cross-site scripting payloads |
| SQL_Injection_Dataset.csv | SQL injection queries |
| master_web_attack_dataset.csv | General web attack payloads (capped for balance) |
| csic_2010.csv | HTTP requests (anomalous/normal) |
| malicious_urls.csv | Malicious and benign URLs (capped for balance) |
| augmented_data.csv | Supplemental data (SSTI, path traversal, benign URLs) |

**After cleaning:**

| Stat | Value |
|------|-------|
| Total Samples | ~695,000 |
| Benign (0) | ~66.7% |
| Malicious (1) | ~33.3% |
| Train / Val / Test Split | 60% / 20% / 20% (stratified) |

## 📁 Project Structure

```
.
├── train_model.py          # Training pipeline (data loading, model, training)
├── test_model.py           # CLI tool to test the trained model
├── show_metrics.py         # Generate all evaluation visualizations
├── gru_model.keras         # Final trained model
├── gru_model_best.keras    # Best model checkpoint (lowest val_loss)
├── tokenizer.pickle        # Fitted character-level tokenizer
├── training_history.json   # Per-epoch training metrics (JSON)
├── epoch_metrics.csv       # Per-epoch metrics (CSVLogger output)
├── test_results.npz        # Test set predictions (for show_metrics.py)
├── metrics/                # Generated evaluation plots
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── class_metrics.png
│   ├── learning_rate.png
│   ├── training_dashboard.png
│   └── epoch_metrics_table.txt
├── requirements.txt        # Python dependencies
├── *.csv                   # Dataset files (see table above)
└── README.md
```

## 🛠️ Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/GRU.git
   cd GRU
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Datasets**
   Place all CSV dataset files in the project root directory. The training script checks for each file and skips any that are missing.

## ▶️ Usage

### Train the Model

```bash
python train_model.py
```

This will:
- Load and combine all available datasets (up to 6 sources)
- Split into 60/20/20 stratified train/val/test sets
- Compute class weights for imbalanced labels
- Train the Bidirectional GRU model with regularization and callbacks
- Save `gru_model.keras`, `gru_model_best.keras`, `tokenizer.pickle`, `training_history.json`, `epoch_metrics.csv`, and `test_results.npz`

### Visualize Metrics

```bash
python show_metrics.py
```

Generates confusion matrix, ROC curve, Precision-Recall curve, training curves, learning rate schedule, training dashboard, and per-class metrics bar chart. All saved in the `metrics/` folder.

### Test the Model

```bash
# Interactive mode — type payloads one at a time
python test_model.py

# Single prediction
python test_model.py --input "<script>alert('XSS')</script>"

# Batch predictions from a file (one payload per line)
python test_model.py --file payloads.txt

# Demo with built-in sample payloads
python test_model.py --demo
```

## 📈 Results

### Confusion Matrix (Held-Out Test Set)

| | Predicted Benign | Predicted Malicious |
|---|---|---|
| **Actual Benign** | 91,721 (TN) | 937 (FP) |
| **Actual Malicious** | 2,033 (FN) | 44,315 (TP) |

### Training Progress

The model trained for 30 epochs with automatic learning rate reductions:

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss | LR |
|-------|-----------|---------|------------|----------|-----|
| 1 | 67.0% | 67.1% | 0.641 | 0.637 | 5e-4 |
| 10 | 94.1% | 95.9% | 0.191 | 0.146 | 2.5e-4 |
| 20 | 97.5% | 97.6% | 0.091 | 0.084 | 1.25e-4 |
| 30 | 97.7% | 97.9% | 0.084 | 0.078 | 6.25e-5 |

## 🔬 Overfitting Analysis

| Check | Result |
|-------|--------|
| Train/Val Loss Gap (final) | 0.006 — negligible |
| Train/Val Accuracy Gap | 0.14% — negligible |
| Val loss trend | Monotonically decreasing across 30 epochs |
| Regularization applied | SpatialDropout (15%), Dense Dropout (20–30%), L2 (1e-5), Class Weights |
| Data leakage | None — tokenizer fit on training set only |
| Evaluation set | Held-out test set, never seen during training or validation |
| **Verdict** | **No overfitting** |

## 💡 Future Work

- **Expand Dataset**: Add command injection and path traversal-specific samples
- **Model Comparison**: Benchmark against LSTMs, Transformers, and 1D-CNNs
- **Hyperparameter Tuning**: Use KerasTuner or Optuna for systematic search
- **Deployment**: REST API with Flask/FastAPI for real-time inference
- **Multi-Class Classification**: Distinguish between attack types (XSS, SQLi, SSTI, etc.)
- **Adversarial Robustness**: Evaluate against evasion techniques and obfuscated payloads

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
