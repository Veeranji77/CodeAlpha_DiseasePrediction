# CodeAlpha_DiseasePrediction

Machine Learning Internship at CodeAlpha - **Task 4: Disease Prediction from Medical Data**

## Objective
Predict whether a tumour is malignant or benign from patient measurements.

## Dataset
Breast Cancer Wisconsin (Diagnostic) from the UCI ML Repository. It is bundled with scikit-learn (`load_breast_cancer`), so nothing needs to be downloaded.
- 569 patients, 30 numeric features, no missing values
- 212 malignant, 357 benign

## Approach
1. Loaded the data and flipped the label so 1 = malignant
2. Basic EDA (class balance, feature correlation)
3. 80/20 stratified train-test split
4. Trained four models: Logistic Regression, SVM, Random Forest, XGBoost
   (scaling applied inside a pipeline for LogReg and SVM)
5. Compared them with 5-fold cross-validation and test metrics

## Results (test set)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| SVM | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9947 |
| Logistic Regression | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 0.9960 |
| Random Forest | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9942 |
| XGBoost | 0.9561 | 1.0000 | 0.8810 | 0.9367 | 0.9937 |

SVM had the best F1. Logistic Regression had the best ROC-AUC. All four are very close, so on a test set of only 114 patients the ranking could change with a different split.
For a disease problem recall is important (a missed malignant case is costly), and it is the weakest metric here at about 0.88 to 0.93.
Most important features (Random Forest): worst perimeter, worst area, worst concave points.

## How to run
```
pip install -r requirements.txt
python disease_prediction.py
```
Plots and the results table are saved in the `outputs/` folder.

## Files
- `disease_prediction.py` - full code
- `outputs/` - confusion matrix, ROC curves, feature importance, model comparison
- `requirements.txt`
