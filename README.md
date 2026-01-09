# Loan Approval – Machine Learning Model Comparison (Flagship)

## Overview
This project predicts loan approval outcomes using supervised machine learning.
I built an end-to-end classification workflow and compared multiple models
across repeated randomized runs to reduce variance from a single train/test split.

## Key Skills Demonstrated
- Data preprocessing (categoricals, scaling, train/test split)
- Model comparison (baseline + tuned models)
- Reliable evaluation (accuracy, precision, ROC-AUC; repeated runs)
- Clear interpretation of results

## Dataset
The dataset contains applicant-level information with a binary target indicating
whether a loan is approved.



## Methodology
1. Load and clean the dataset
2. Encode categorical features and scale numeric features where appropriate
3. Train and evaluate multiple models:
   - Logistic Regression
   - Decision Tree
   - K-Nearest Neighbors (KNN)
   - Random Forest
4. Repeat the experiment across **N randomized runs** to obtain stable averages
5. Apply hyperparameter tuning (GridSearchCV) to improve generalization

## Results (Summary)
| Model | Mean Accuracy | Mean Precision | Mean ROC-AUC |
|------|---------------|----------------|--------------|
| Logistic Regression | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD |
| KNN | TBD | TBD | TBD |
| Random Forest (Tuned) | TBD | TBD | TBD |

**Best model:** TBD  
**Why it won:** TBD (1–2 sentence interpretation)

## How to Run
1. Create an environment and install dependencies:
   ```bash
   pip install -r requirements.txt
