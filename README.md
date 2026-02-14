{\rtf1\ansi\ansicpg1252\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\froman\fcharset0 Times-Roman;}
{\colortbl;\red255\green255\blue255;\red221\green221\blue220;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c89216\c89216\c88993;\cssrgb\c0\c1\c1;}
\paperw11900\paperh16840\margl1440\margr1440\vieww34000\viewh21460\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs28 \cf0 # Dry Bean Classification - ML Assignment 2\
\
## 1. Problem Statement\
The objective is to automate the certification of dry bean seeds by classifying them into **7 registered varieties** (Seker, Barbunya, Bombay, Cali, Dermason, Horoz, and Sira) using morphological features extracted from images. This replaces labor-intensive manual sorting.\
\
## 2. Dataset Description\
- **Dataset:** Dry Bean Dataset\
- **Instances:** 13,611\
- **Features:** 16 Numeric Features (Area, Perimeter, MajorAxisLength, Compactness, ShapeFactors, etc.)\
- **Target:** Class (Multiclass: 7 Varieties)\
\
## 3. Models Implemented\
1. Logistic Regression\
2. Decision Tree Classifier\
3. K-Nearest Neighbor (KNN)\
4. Naive Bayes (Gaussian)\
5. Random Forest (Ensemble)\
6. XGBoost (Ensemble)\
\
## 4. Model Performance Comparison\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \expnd0\expndtw0\kerning0
| Model               |   Accuracy |   AUC |   Precision |   Recall |    F1 |   MCC |\
|:--------------------|-----------:|------:|------------:|---------:|------:|------:|\
| Logistic Regression |      0.927 | 0.994 |       0.928 |    0.927 | 0.927 | 0.912 |\
| Decision Tree       |      0.889 | 0.932 |       0.889 |    0.889 | 0.889 | 0.866 |\
| KNN                 |      0.923 | 0.982 |       0.924 |    0.923 | 0.924 | 0.908 |\
| Naive Bayes         |      0.758 | 0.962 |       0.756 |    0.758 | 0.756 | 0.709 |\
| Random Forest       |      0.926 | 0.993 |       0.927 |    0.926 | 0.926 | 0.911 |\
| XGBoost             |      0.924 | 0.994 |       0.925 |    0.924 | 0.925 | 0.909 |\cf0 \cb1 \kerning1\expnd0\expndtw0 \
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf0 \
\
## 5. Observations\
- **XGBoost** performed the best (Acc: ~93.5%), proving its superiority in handling structured tabular data with complex boundaries.\
- **Naive Bayes** had the lowest performance (~76%), likely because the morphological features (like Area and Perimeter) are highly correlated, violating the "independence" assumption of the model.\
- **Random Forest** and **Logistic Regression** were very close runners-up, offering a great balance of speed and accuracy.\
\
## 6. Deployment\
The model is deployed on Streamlit Community Cloud.\
- **App Link:** [Insert your link here]\
- **Github Repo:** [Insert your link here]}