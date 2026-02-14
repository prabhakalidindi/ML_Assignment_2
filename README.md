# Dry Bean Classification - ML Assignment 2

## 1. Problem Statement
The objective is to automate the certification of dry bean seeds by classifying them into **7 registered varieties** (Seker, Barbunya, Bombay, Cali, Dermason, Horoz, and Sira) using morphological features extracted from images. This replaces labor-intensive manual sorting.

## 2. Dataset Description
- **Dataset:** Dry Bean Dataset
- **Instances:** 13,611
- **Features:** 16 Numeric Features (Area, Perimeter, MajorAxisLength, Compactness, ShapeFactors, etc.)
- **Target:** Class (Multiclass: 7 Varieties)

## 3. Models Implemented
The following 6 classification models were implemented and compared:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

## 4. Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|:---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.927** | **0.994** | **0.928** | **0.927** | **0.927** | **0.912** |
| Decision Tree | 0.889 | 0.932 | 0.889 | 0.889 | 0.889 | 0.866 |
| KNN | 0.923 | 0.982 | 0.924 | 0.923 | 0.924 | 0.908 |
| Naive Bayes | 0.758 | 0.962 | 0.756 | 0.758 | 0.756 | 0.709 |
| Random Forest | 0.926 | 0.993 | 0.927 | 0.926 | 0.926 | 0.911 |
| XGBoost | 0.924 | 0.994 | 0.925 | 0.924 | 0.925 | 0.909 |

## 5. Observations
- **Logistic Regression** and **Random Forest** achieved the highest accuracy (~92.7%), proving that both linear and ensemble methods are effective for this structured dataset.
- **XGBoost** followed closely with ~92.4% accuracy, showing strong capability in handling complex boundaries.
- **Naive Bayes** had the lowest performance (~75.8%), likely because the morphological features (like Area and Perimeter) are highly correlated, which violates the model's "independence" assumption.
- **Decision Tree** showed lower accuracy (~88.9%) compared to the ensemble methods, suggesting it may have slightly overfitted the training data.

## 6. Deployment
The model is deployed on Streamlit Community Cloud
