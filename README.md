# Credit Card Fraud Detection

End-to-end machine learning notebook for detecting fraudulent credit card transactions on a highly imbalanced dataset (0.17% fraud rate). Built as part of the **Live Pakistan — Machine Learning Internship, Week 2**.

## Problem

Out of 284,807 transactions, only ~492 are fraud. A naive model can achieve 99.8% accuracy while catching **zero** fraud. This project avoids that trap by focusing on the right metrics, imbalance strategies, and business-driven threshold selection.

## What this notebook covers

1. **Class imbalance analysis** — why accuracy is misleading here
2. **Resampling comparison** — class weighting, SMOTE, ADASYN, SMOTEENN, SMOTETomek
3. **Supervised models** — Logistic Regression, Random Forest, XGBoost
4. **Unsupervised baselines** — Isolation Forest, Local Outlier Factor
5. **Cost-sensitive thresholding** — optimize for business cost, not default 0.5
6. **Explainability** — SHAP summary plots (with permutation-importance fallback)
7. **Concept drift check** — monitor performance over time windows
8. **Live stream simulation** — score transactions one-by-one and log flagged cases
9. **Model export** — save a reusable pipeline artifact with scaler, features, and threshold

## Key results

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **XGBoost** (best) | 0.697 | 0.800 | 0.745 | — | **0.818** |
| Random Forest | — | — | — | — | 0.795 |
| Logistic Regression | — | — | — | — | lower |

- **Best resampling strategy:** SMOTETomek (hybrid oversample + boundary cleanup)
- **Cost-optimal threshold:** 0.12 (vs default 0.5) — minimizes total business cost at $500/missed fraud and $10/false alarm
- **At threshold 0.12:** 129 false positives, 15 missed frauds, total cost $8,790

## Project structure

```
Credit_Card_Fraud_Detection/
├── Credit_Card_Fraud_Detection.ipynb   # Main notebook
├── requirements.txt                    # Python dependencies
├── flagged_transactions_log.csv        # Sample output from live-stream simulation
├── creditcard.csv                      # Dataset (not in repo — see setup below)
└── fraud_detection_pipeline.pkl        # Saved model (generated after running notebook)
```

## Setup

### 1. Clone the repository

```powershell
git clone https://github.com/<your-username>/Credit_Card_Fraud_Detection.git
cd Credit_Card_Fraud_Detection
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Download the dataset

Download `creditcard.csv` from Kaggle and place it in the project root:

[Kaggle — Credit Card Fraud Detection (ULB ML Group)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

> The dataset is ~150 MB and is excluded from this repo via `.gitignore`.

### 4. Run the notebook

Open and run all cells in `Credit_Card_Fraud_Detection.ipynb`:

```powershell
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

## Generated artifacts (after running)

| File | Description |
|---|---|
| `fraud_detection_pipeline.pkl` | Trained model + scaler + feature columns + decision threshold |
| `class_imbalance.png` | Fraud vs genuine class distribution |
| `precision_recall_curve.png` | PR curve with best-F1 threshold |
| `cost_sensitive_threshold.png` | Business cost vs threshold |
| `shap_summary.png` | SHAP feature importance (or `permutation_importance.png` fallback) |
| `flagged_transactions_log.csv` | Simulated real-time flagged transactions |

## Tech stack

- **Python 3** · pandas · NumPy · scikit-learn · imbalanced-learn · XGBoost · SHAP · Matplotlib · Seaborn

## Dataset citation

> Dal Pozzolo, A. Cao, F. L. Angiola, O. Bontempi, G. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE.

## License

This project is for educational purposes as part of the ML Internship program.
