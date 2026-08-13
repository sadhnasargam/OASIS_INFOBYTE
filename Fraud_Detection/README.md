# Fraud Detection using Machine Learning

## Project Overview
This project focuses on detecting fraudulent credit card transactions using Machine Learning techniques. Since fraud datasets are highly imbalanced, SMOTE (Synthetic Minority Oversampling Technique) was used to balance the classes before training the models.

The project includes Exploratory Data Analysis (EDA), class imbalance handling, model training, performance evaluation, feature importance analysis, and scalability discussion.

## Objective
- Detect fraudulent financial transactions.
- Handle class imbalance using SMOTE.
- Compare the performance of Logistic Regression and Random Forest models.
- Evaluate the models using Precision, Recall, F1-Score, and ROC-AUC.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Joblib

## Dataset
- Dataset: Credit Card Fraud Detection
- Source: Kaggle
- File: `creditcard.csv`
### Dataset Note:
**Due to GitHub file size limitation, the dataset is not included in thi repository . Ypu can download it from Kaggle:` https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud%E2%81%A0`

## Exploratory Data Analysis (EDA)

### 1. Fraud vs Non-Fraud Transactions
![alt text](<Fraud vs Non-Fraud Transactions.png>)

### 2. Transaction Amount Distribution
![alt text](<Transaction Amount Distribution.png>)

### 3. Time-of-Day Analysis
![alt text](<Time-of-Day Analysis.png>)

### 4. Correlation Heatmap
![alt text](<Correlation Heatmap.png>)

## Class Imbalance Handling

The dataset contains very few fraudulent transactions compared to normal transactions.

To solve this problem, **SMOTE (Synthetic Minority Oversampling Technique)** was applied to balance the training data before model training.

## Machine Learning Models

### Logistic Regression

### Random Forest Classifier

## Model Evaluation

The following metrics were used:

- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

## ROC Curve
![alt text](<ROC Curve.png>)

## Feature Importance
![alt text](<Feature Importance.png>)

## Results

- Successfully detected fraudulent transactions.
- SMOTE improved class balance.
- Random Forest performed better than Logistic Regression.
- ROC-AUC, Precision, Recall and F1-Score were used instead of Accuracy because the dataset is highly imbalanced.

## Scalability Discussion

This solution can be scaled to process millions of transactions per hour by using distributed frameworks such as Apache Spark and Apache Kafka. The trained model can be deployed with Flask or FastAPI and hosted on cloud platforms like AWS, Azure, or Google Cloud for real-time fraud detection.

## Author

**Sadhna Kumari**