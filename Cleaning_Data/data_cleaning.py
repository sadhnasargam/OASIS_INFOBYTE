import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Titanic-Dataset.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.info())
print(df.isnull().sum())
print(df.describe())

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin"])
print("missing value after cleaning")
print(df.isnull().sum())
print("Duplicate Row Before Removing: ")
print(df.duplicated().sum())

df = df.drop_duplicates()
print("Duplicate Row After Removing: ")
print(df.duplicated().sum())

print("Data Type: ")
print(df.dtypes)

df["Sex"] = df["Sex"].str.capitalize()
df["Embarked"] = df["Embarked"].str.upper()
print(df.head())

Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
print("Lower Limit: ", lower_limit)
print("Upper Limit: ", upper_limit)

plt.figure(figsize=(8,5))
sns.boxplot(x=df["Fare"])
plt.title("Box Plot of Fare")
plt.show()

df = df[(df["Fare"] >= lower_limit) & (df["Fare"] <= upper_limit)]
print("New Shape: ", df.shape)
sns.boxplot(x=df["Fare"])
plt.title("Box Plot After Removing Outliers")
plt.show()

df.to_csv("Titanic_Cleaned.csv", index=False)
print("Cleaned dataset saved successfully")

