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

## Results (Aggregated over 10 runs)

Model performance was evaluated across 10 randomized train/test splits.
The table below reports mean performance metrics to ensure stability and
reduce dependence on a single split.

| Model        | Mean Accuracy | Std Accuracy | Mean Precision | Mean ROC-AUC |
|--------------|---------------|--------------|----------------|--------------|
| DT (Tuned)   | **0.9801**    | 0.0055       | **0.9849**     | **0.9861**   |
| DT           | 0.9788        | 0.0050       | 0.9849         | 0.9781       |
| LR (Tuned)   | 0.9352        | 0.0054       | 0.9673         | 0.9651       |
| KNN (Tuned)  | 0.9218        | 0.0087       | 0.9359         | 0.9749       |
| LR           | 0.9153        | 0.0052       | 0.9300         | 0.9681       |
| KNN          | 0.9128        | 0.0096       | 0.9333         | 0.9697       |

**Best model:** Tuned Decision Tree (DT_Tuned)  
**Selection criterion:** Highest mean accuracy across 10 runs  
**Final performance:**  
- Mean Accuracy = **0.9801**  
- Mean Precision = **0.9849**  
- Mean ROC-AUC = **0.9861**

## Discussion

Across all evaluated models, tree-based approaches consistently outperformed
linear and distance-based classifiers. The tuned Decision Tree achieved the
highest mean accuracy and ROC-AUC while maintaining low variance across runs,
indicating strong generalization performance.

Hyperparameter tuning improved performance for all models, with the largest
gains observed in the Decision Tree and Logistic Regression classifiers.
Repeated evaluation over multiple randomized splits provided more reliable
performance estimates than a single train/test split and reduced sensitivity
to random sampling effects.

These results suggest that non-linear decision boundaries are well-suited
for this dataset and that model tuning plays a critical role in achieving
optimal predictive performance.


## Conclusion

This project demonstrates the importance of systematic model comparison,
robust evaluation metrics, and hyperparameter tuning in financial decision
support tasks. Among the tested models, the tuned Decision Tree provided the
best balance of accuracy, precision, and ROC-AUC, making it the most reliable
choice for loan approval prediction in this dataset.

The use of repeated randomized runs strengthened confidence in the reported
results and highlights best practices for applied machine learning workflows.

## How to Run
1. Create an environment and install dependencies:
   ```bash
   pip install -r requirements.txt
