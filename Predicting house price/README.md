# House Price Prediction

A Machine Learning project that predicts house prices using property-related features such as location, carpet area, property status, floor, furnishing, bathrooms, balconies, parking, ownership, and super area.

The project covers data preprocessing, exploratory data analysis, feature engineering, categorical encoding, model training, evaluation, and model saving using Joblib.

---

## Project Overview

House Price Prediction is a regression-based Machine Learning project developed to estimate property prices from historical real-estate data.

The dataset contains **187,531 property records with 21 features**.

After data cleaning and removing records with missing or invalid prices, the data was prepared for Machine Learning.

### Target Variable

`Price (in rupees)`

### Machine Learning Algorithm

**Linear Regression**

---

## Objectives

- Analyze the real-estate dataset
- Handle missing and invalid data
- Convert property-related features into suitable numerical formats
- Perform Exploratory Data Analysis (EDA)
- Encode categorical variables using One-Hot Encoding
- Train a Linear Regression model
- Evaluate model performance
- Analyze model coefficients
- Save the trained model and preprocessing pipeline using Joblib

---

## Dataset Information

The dataset contains information about residential properties.

### Important Features

| Feature | Description |
|---|---|
| `location` | Location of the property |
| `Carpet Area` | Carpet area of the property |
| `Status` | Current property status |
| `Floor` | Floor information |
| `Transaction` | Type of transaction |
| `Furnishing` | Furnishing status |
| `facing` | Direction the property faces |
| `overlooking` | Property overlooking information |
| `Bathroom` | Number of bathrooms |
| `Balcony` | Number of balconies |
| `Car Parking` | Parking information |
| `Ownership` | Ownership type |
| `Super Area` | Super area of the property |

### Target

`Price (in rupees)`

---
### Data Preprocessing

The following preprocessing steps were performed:
1. Handling Missing Target Values: 
Rows with missing values in Price (in rupees) were removed.

2. Removing Invalid Prices:
Records with zero or negative property prices were removed.

3. Converting Area Features:
The following columns were converted into numerical values:

Carpet Area
Super Area
Bathroom
Balcony
4. Removing Unnecessary Features:
The following columns were excluded from model training:

Index
Title
Description
Amount(in rupees)
Society
Dimensions
Plot Area

5. Missing Value Imputation:
Numerical features were handled using median imputation.
Categorical features were handled using most-frequent imputation.

6. Categorical Encoding:
Categorical features were converted into numerical features using:

OneHotEncoder(
    drop='first',
    handle_unknown='ignore'
)
### Exploratory Data Analysis

1. House Price Distribution:
A histogram with KDE was used to understand the distribution of property prices.

- **House Price Distribution graph**

![alt text](<House Price Distribution.png>)

2. Correlation Heatmap:
- A correlation heatmap was created to analyze relationships between numerical variables.

**Correlation Heatmap Graph**

![alt text](<Correlation Heatmap.png>)

## Model Training

The project uses Linear Regression for house price prediction.

The dataset was divided into:

**Training Data:-**80% 
**Testing Data:-**20% 

The preprocessing pipeline was fitted only on the training data to avoid data leakage.

## Model Evaluation

The trained model was evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

MSE	- 9,343,440.10
RMSE - 3,056.70
R² Score -	0.6075

## Model Performance

The model achieved an R² Score of 0.6075, meaning it explains approximately 60.75% of the variation in house prices on the test dataset.

1. Actual vs Predicted Prices

The Actual vs Predicted plot compares the actual house prices with the prices predicted by the Linear Regression model.


**Actual vs Predicted graph**

![alt text](<Actual vs Predicted.png>)

2. Residual Analysis

Residual analysis was performed to understand the difference between actual and predicted prices.
Residual = Actual Price - Predicted Price

**Residual Plot Graph**

![alt text](<Residual Plot.png>)


## Feature Coefficient Analysis

The Linear Regression coefficients were analyzed to understand the relationship between encoded features and predicted house prices.

### Top Positive Coefficients
```
Some of the strongest positive coefficients included:

location_mumbai
Floor_1 out of 1
location_navi-mumbai
location_new-delhi
location_thane
```
### Top Negative Coefficients
```
Some of the strongest negative coefficients included:

location_bhiwadi
location_sonipat
Car Parking_10 Open
Floor_14 out of 15
location_raipur

Coefficients represent the effect of individual encoded features relative to their reference categories.
```
## Model Saving

The trained model and preprocessing pipeline were saved using Joblib.

joblib.dump(model, "house_price_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

These files can be loaded later to make predictions without retraining the model.

## Dataset Note

**Due to GitHub file size limitation, the dataset is not included in thi repository . Ypu can download it from Kaggle: `https://www.kaggle.com/datasets/juhibhojani/house-price`**

## Technologies & Libraries
### Programming Language
```
Python
Pandas
NumPy
Matplotlib
Seaborn
Machine Learning
Scikit-learn
Linear Regression
One-Hot Encoding
Joblib
```

## How to Run the Project
```
1. Open the Project Folder

cd Predicting_House_Prices

2. Install Dependencies

pip install -r requirements.txt

3. Run the Project

python demo.py
```
## Requirements

The project requires the following Python libraries:
```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
joblib
```
## Future Improvements

The model can be further improved by:

Trying advanced regression algorithms
Comparing Linear Regression with Random Forest and Gradient Boosting
Performing hyperparameter tuning
Creating additional location-based features
Handling extreme price outliers
Performing feature selection
Deploying the model using Streamlit or Flask

## Author

**Sadhna Kumari**

