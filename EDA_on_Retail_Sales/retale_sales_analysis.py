import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("retail_sales_dataset.csv")
print(df.head())
print("missing values: ")
print(df.isnull().sum())
print("Duplicate Rows: ", df.duplicated().sum())
print(df.describe())
print("Mean: ")
print(df.mean(numeric_only=True))
print("Median: ")
print(df.median(numeric_only=True))
print("Mode: ")
print(df.mode().iloc[0])
print("Standerd Deviation: ")
print(df.std(numeric_only=True))
print(df.dtypes)

# Here we convert date into datetime format
df["Date"]= pd.to_datetime(df["Date"],dayfirst=True)
print(df["Date"].dtype)

# Here we made a month and Quarter column
df["Month"]= df["Date"].dt.month_name()
df["Quarter"]= df["Date"].dt.quarter
print(df.head())

# Here we made a monthly sales trend graph
monthly_sales = df.groupby("Month")["Total Amount"].sum()
print(monthly_sales)
monthly_sales.plot(kind="line", marker="o",figsize=(11,6))
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.grid(True)
plt.show()

# Here we made a Quarterly Sales Trend Graph
quarterly_sales = df.groupby("Quarter")["Total Amount"].sum()
print(quarterly_sales)
quarterly_sales.plot(kind="bar",figsize=(8,5))
plt.title("Quarterly Sales Trend")
plt.xlabel("Quarterly")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.show()

# Gender Distribution Pie Chart
gender_count = df["Gender"].value_counts()
print(gender_count)
gender_count.plot(kind="pie", autopct="%1.1f%%",figsize=(6,6))
plt.title("Gender Distribution")
plt.ylabel("")
plt.show()

# Age Distribution Histo Graph
plt.figure(figsize=(8,5))
plt.hist(df["Age"], bins=10)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.grid(True)
plt.show()

# Total Sales by Product Category
category_sales = df.groupby("Product Category")["Total Amount"].sum()
print(category_sales)
plt.figure(figsize=(8,5))
category_sales.plot(kind="bar")
plt.title("Total Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.show()

# Quantity Sold by Product Category
category_quantity = df.groupby("Product Category")["Quantity"].sum()
print(category_quantity)
plt.figure(figsize=(8,5))
category_quantity.plot(kind="bar")
plt.title("Quantity Sold by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.show()

# Correlation
correlation = df[["Age", "Quantity", "Price per Unit", "Total Amount"]].corr()
print(correlation)
plt.figure(figsize=(8,6))
sns.heatmap(correlation, annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

# Average Purchase Amount by Product Category
average_sales = df.groupby("Product Category")["Total Amount"].mean()
print(average_sales)
average_sales.plot(kind="bar", figsize=(8,5))
plt.title("Average Purchase Amount by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Average Purchase Amount")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.show()
