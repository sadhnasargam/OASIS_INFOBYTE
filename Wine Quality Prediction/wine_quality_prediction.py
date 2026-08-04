import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv("winequality-red.csv", sep=",")
print(df.head())
print(df.info())
print(df.describe())
print("\nMissing Value: ")
print(df.isnull().sum())

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True,cmap="coolwarm")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="quality", data=df)
plt.title("Wine Quality Distribution")
plt.xlabel("Quality")
plt.ylabel("Count")
plt.show()

from sklearn.model_selection import train_test_split
X = df.drop("quality", axis=1)
y = df["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)
print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color="blue")
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],color="red", linewidth=2)

plt.xlabel("Actual Wine Quality")
plt.ylabel("Predicted Wine Quality")
plt.title("Actual vs Predicted Wine Quality")
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(6,6))
plt.scatter(y_pred, residuals, color="green")
plt.axhline(y=0, color="red", linestyle="--")

plt.xlabel("Predicted Quality")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

sample = X_test.iloc[[0]]
prediction = model.predict(sample)
print("Actual Quality:", y_test.iloc[0])
print("Predicted Quality:", prediction[0])

joblib.dump(model, "wine_quality_model.pkl")
print("Model saved successfully!")