import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df= pd.read_csv("house_prices.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print((df.isnull().sum()/len(df))*100)

# Remove rows where target price is missing
df = df.dropna(subset=['Price (in rupees)'])
df = df[df['Price (in rupees)'] > 0].copy()
print("\nData after removing invalid prices:")
print(df.shape)
print("\nMissing values after removing missing target:")
print(df.isnull().sum())

# Distribution of House Prices
plt.figure(figsize=(8, 6))
sns.histplot(df['Price (in rupees)'], kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price (in rupees)')
plt.ylabel('Frequency')
plt.show()

# Convert area-related columns to numeric values
df['Carpet Area'] = pd.to_numeric(df['Carpet Area'].astype(str).str.extract(r'([\d.]+)')[0],errors='coerce')
df['Super Area'] = pd.to_numeric(df['Super Area'].astype(str).str.extract(r'([\d.]+)')[0], errors='coerce')
df['Bathroom'] = pd.to_numeric(df['Bathroom'].astype(str).str.extract(r'(\d+\.?\d*)')[0],errors='coerce')
df['Balcony'] = pd.to_numeric(df['Balcony'].astype(str).str.extract(r'(\d+\.?\d*)')[0],errors='coerce')

print("\nData types after conversion:")
print(df[['Carpet Area', 'Super Area', 'Bathroom', 'Balcony']].dtypes)
print("\nSample values after conversion:")
print(df[['Carpet Area', 'Super Area', 'Bathroom', 'Balcony']].head(10))

# Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr,annot=True,cmap='coolwarm',fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Separate features and target
X = df.drop('Price (in rupees)', axis=1)
y = np.log1p(df['Price (in rupees)'])

# Remove columns that are not useful for prediction
columns_to_drop = ['Index','Title','Description','Amount(in rupees)','Society','Dimensions','Plot Area']
X = X.drop(columns=columns_to_drop, errors='ignore')
print("\nSelected Features:")
print(X.columns.tolist())

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(include='object').columns

# Numerical preprocessing
numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=True))])

# Combine numerical and categorical preprocessing
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols),('cat', categorical_transformer, categorical_cols)])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# Fit preprocessing only on training data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
print("\nEncoded Feature Matrix Shape:")
print(X_train.shape)
print("\nPreprocessing and Encoding completed successfully!")

# Train Linear Regression Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Evaluation
from sklearn.metrics import mean_squared_error, r2_score
y_pred_original = np.expm1(y_pred)
y_test_original = np.expm1(y_test)

mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred_original)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


# Get actual feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()
coefficients = model.coef_
coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients})

# Sort coefficients from highest to lowest
coef_df = coef_df.sort_values(by="Coefficient", ascending=False)
print("\nTop 10 Positive Coefficients:")
print(coef_df.head(10))

print("\nTop 10 Negative Coefficients:")
print(coef_df.tail(10))

# Save the trained model
joblib.dump(model, "house_price_model.pkl")

# Save the preprocessing pipeline
joblib.dump(preprocessor, "house_price_preprocessor.pkl")
print("\nModel and preprocessor saved successfully!")