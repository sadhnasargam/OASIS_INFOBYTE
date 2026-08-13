# Google Play Store Analysis

## Project Overview

This project performs an in-depth Exploratory Data Analysis (EDA) of the Google Play Store ecosystem using Python. The analysis focuses on cleaning real-world app data, exploring app categories, analyzing ratings, installs, pricing trends, and performing sentiment analysis on user reviews to generate meaningful business insights for app developers.

## Objective

- Clean and preprocess Google Play Store datasets.
- Analyze app categories and ratings.
- Study installs, app size, and pricing trends.
- Perform sentiment analysis on user reviews.
- Generate business insights using visualizations.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- TextBlob
- Plotly
- Jupyter Notebook / VS Code

## Dataset

This project uses two datasets:

1. **Google Play Store Apps Dataset**
2. **Google Play Store User Reviews Dataset**

Source:
https://www.kaggle.com/datasets/lava18/google-play-store-apps

## Features Implemented

### Data Loading
- Loaded Google Play Store Apps dataset
- Loaded User Reviews dataset

### Data Cleaning
- Removed duplicate records
- Handled missing values
- Converted incorrect data types
- Cleaned Installs column
- Cleaned Price column
- Converted Size into numeric values

### Exploratory Data Analysis (EDA)

- App Category Distribution
- Rating Distribution
- Average Rating by Category
- App Size vs Installs Analysis
- Free vs Paid Apps Comparison
- Paid App Price Distribution
- Revenue Estimation by Category

### Sentiment Analysis

- Positive Reviews
- Neutral Reviews
- Negative Reviews

Performed using **TextBlob**.

### Category-wise Sentiment Analysis

Analyzed positive, neutral, and negative sentiments across different app categories.

### Interactive Visualization

Created an interactive Plotly visualization for better exploration of the dataset.

# Visualizations

## 1. Category Distribution
![alt text](<Category Distribution.png>)

## 2. Rating Distribution

![alt text](<Rating Distribution.png>)

## 3. Average Rating by Category

![alt text](<Average Rating by Category.png>)

## 4. Size vs Installs

![alt text](<Size vs Installs.png>)

## 5. Free vs Paid Apps

![alt text](<Free vs Paid Apps.png>)

## 6. Paid App Price Distribution

![alt text](<Paid App Price Distribution.png>)

## 7. Sentiment Distribution

![alt text](<Sentiment Distribution.png>)

## 8. Sentiment by Category

![alt text](<Sentiment by Category.png>)

## Key Insights

- Most applications on the Play Store are free.
- Family, Games, and Tools are the most competitive categories.
- Most apps maintain ratings above 4.0.
- Paid apps represent a very small portion of the marketplace.
- Positive user reviews significantly outnumber negative reviews.
- Categories with higher installs generally receive more user engagement.

---

## Future Improvements

- Build an app recommendation system.
- Predict app ratings using Machine Learning.
- Deploy the project using Streamlit.
- Create an interactive dashboard using Power BI.

## How to Run

Go to the project folder
```bash
cd Google-Play-Store-Analysis
```
Install dependencies
```bash
pip install -r requirements.txt
```
Run the project
```bash
python playstore_analysis.py
```

## Author

**Sadhna Kumari**