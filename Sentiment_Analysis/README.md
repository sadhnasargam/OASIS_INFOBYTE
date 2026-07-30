# Twitter Sentiment Analysis using Machine Learning

## Project Overview

This project focuses on analyzing the sentiment of Twitter posts using **Natural Language Processing (NLP)** and **Machine Learning**. The objective is to classify tweets as **Positive** or **Negative** based on their text content.

The project includes data preprocessing, text cleaning, feature extraction using TF-IDF, model training using Logistic Regression, and performance evaluation.

# Objective

The main objectives of this project are:

- Analyze Twitter text data.
- Clean and preprocess raw tweets.
- Convert text into numerical features using TF-IDF.
- Train a Machine Learning model for sentiment prediction.
- Predict whether a tweet is Positive or Negative.
- Evaluate the model using different performance metrics.

# Dataset

**Dataset Name:** Dataset

The dataset contains thousands of Twitter posts collected for sentiment classification.

### Dataset Features

- Target
- ID
- Date
- Flag
- User
- Tweet Text

### Target Labels

- **0 → Negative Sentiment**
- **4 → Positive Sentiment**

During preprocessing, the positive label (4) was converted into **1**.

# Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- VS Code

# Machine Learning Algorithm

**Logistic Regression**

Logistic Regression is a supervised machine learning algorithm used for binary classification problems. In this project, it is used to classify tweets into Positive and Negative sentiment.

# Project Workflow

## Step 1: Import Required Libraries

Imported all required Python libraries such as Pandas, NumPy, NLTK, Matplotlib, and Scikit-learn.

---

## Step 2: Load Dataset

Loaded the dataset into a Pandas DataFrame.

---

## Step 3: Select Required Columns

Selected only the important columns required for sentiment analysis.

- Target
- Tweet Text

---

## Step 4: Data Preprocessing

Performed several preprocessing tasks:

- Converted Positive label (4) into 1
- Removed unwanted columns
- Checked missing values
- Selected useful data

---

## Step 5: Text Cleaning

Cleaned tweet text by:

- Converting text into lowercase
- Removing URLs
- Removing punctuation
- Removing numbers
- Removing special characters
- Removing extra spaces

---

## Step 6: Use NLTK

using NLTK to improve model performance.

Examples:

- the
- is
- am
- are
- and
- of
- in
- to

---

## Step 7: Feature Extraction

Converted cleaned text into numerical vectors using **TF-IDF Vectorizer**.

TF-IDF helps the machine understand the importance of words in the tweets.

---

## Step 8: Split Dataset

Split the dataset into:

- Training Data (80%)
- Testing Data (20%)

---

## Step 9: Train Machine Learning Model

Trained the Logistic Regression model using the training dataset.

---

## Step 10: Predict Sentiment

Predicted sentiments for the testing dataset.

---

## Step 11: Evaluate Model

Evaluated the model using:

- Accuracy Score
- Precision
- Recall
- F1 Score
- Classification Report

---

# Results

The model successfully classified tweets into Positive and Negative sentiments.

### Model Accuracy

**Accuracy: 74.88%**

The model achieved good performance in predicting tweet sentiments.

---

# Output Screenshots

## 1 Dataset Preview
![alt text](<Dataset Preview.png>)

## 2 Dataset Informetion
![alt text](<Dataset Informetion.png>)

## 3 Cleaned Tweet Text
![alt text](<Cleaned Tweet.png>)

## 5 Sentiment Distribution Graph
![alt text](<Sentiment Distribution Graph.png>)

## Model Accuracy Output
![alt text](<Model Accuracy.png>)

# Future Improvements

This project can be improved by:

- Using larger datasets
- Applying Deep Learning models
- Using LSTM or BERT
- Performing Hyperparameter Tuning
- Improving text preprocessing

# Conclusion

This project successfully demonstrates how Natural Language Processing and Machine Learning can be used to analyze public opinion from Twitter data.

The tweets were cleaned, converted into numerical features using TF-IDF, and classified using Logistic Regression. The model achieved approximately **75% accuracy**, making it effective for basic sentiment prediction tasks.

This project helped in understanding text preprocessing, feature engineering, machine learning model training, and model evaluation.


# Author

**Sadhna Kumari**