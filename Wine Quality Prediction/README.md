# Wine Quality Prediction

## Project Overview
This project aims to predict the quality of red wine using Machine Learning based on its physicochemical properties. The dataset was analyzed, cleaned, visualized, and used to train a regression model for predicting wine quality.

## Objective
- Analyze the wine quality dataset.
- Perform Exploratory Data Analysis (EDA).
- Build a Machine Learning model to predict wine quality.
- Evaluate model performance using regression metrics.

## Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Dataset
- **Dataset:** Wine Quality Dataset (Red Wine)
- **Records:** 1599
- **Features:** 11 Input Features + 1 Target Variable (Quality)

### Input Features
- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

### Target
- Wine Quality

## Exploratory Data Analysis
The following analyses were performed:

- Dataset Inspection
- Missing Value Analysis
- Statistical Summary
- Correlation Heatmap
- Wine Quality Distribution

## Machine Learning Model
A Regression Model was trained to predict wine quality.

### Train-Test Split
- Training Data: 1279 Samples
- Testing Data: 320 Samples

## Model Evaluation

Performance Metrics:

- Mean Squared Error (MSE): **0.3900**
- Root Mean Squared Error (RMSE): **0.6245**
- R² Score: **0.4031**

Sample Prediction:

- Actual Quality: **6**
- Predicted Quality: **5.35**

## Visualizations

The project includes:

- Correlation Heatmap
![alt text](<Correlation Heatmap.png>)

- Wine Quality Distribution
![alt text](<Wine Quality Distribution.png>)

- Actual vs Predicted Plot
![alt text](<Actual vs Predicted Plot.png>)

- Residual Plot
![alt text](<Residual Plot.png>)

## Model Saving

The trained model is saved using Joblib.

```python
joblib.dump(model, "wine_quality_model.pkl")
```

## Future Improvements

- Improve prediction accuracy using advanced regression models.
- Hyperparameter tuning.
- Feature engineering.
- Deploy the model using Flask or Streamlit.

## Conclusion

This project demonstrates the complete Machine Learning workflow, including data preprocessing, exploratory data analysis, model training, evaluation, prediction, and model saving. It provides practical experience in regression-based predictive analytics using Python.

## Author

**Sadhna Kumari**
