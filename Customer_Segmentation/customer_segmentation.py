import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv("Mall_Customers.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.columns)
print("shape: ",df.shape)
print(df.describe())
print(df.describe(include="object"))

gender_count = df["Gender"].value_counts()
print(gender_count)
plt.figure(figsize=(6,6))
gender_count.plot(kind="pie",autopct="%1.1f%%")
plt.title("Gender Distribution")
plt.ylabel("")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df["Age"],bins=10)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df["Annual Income (k$)"],bins=10)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Number of Customers")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df["Spending Score (1-100)"],bins=10)
plt.title("Spending Score Distribution")
plt.xlabel("Spending Score")
plt.ylabel("Number of Customers")
plt.grid(True)
plt.show()

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
print(X.head())
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[:5])

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeams = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeams.fit(X_scaled)
    wcss.append(kmeams.inertia_)
print(wcss)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

kmeams = KMeans(n_clusters=5, random_state=42, n_init=10)
y_kmeans = kmeams.fit_predict(X_scaled)
df["Cluster"] = y_kmeans
print(y_kmeans)
print(df.head())   

# Customer Segmentation Scatter Plot
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[y_kmeans == 0, 0], X_scaled[y_kmeans == 0, 1], label="Cluster 1")
plt.scatter(X_scaled[y_kmeans == 1, 0], X_scaled[y_kmeans == 1, 1], label="Cluster 2")
plt.scatter(X_scaled[y_kmeans == 2, 0], X_scaled[y_kmeans == 2, 1], label="Cluster 3")
plt.scatter(X_scaled[y_kmeans == 3, 0], X_scaled[y_kmeans == 3, 1], label="Cluster 4")
plt.scatter(X_scaled[y_kmeans == 4, 0], X_scaled[y_kmeans == 4, 1], label="Cluster 5")
plt.scatter(kmeams.cluster_centers_[:,0],kmeams.cluster_centers_[:,1],marker="X",s=200, label="Centroids")
plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")
plt.legend()
plt.grid(True)
plt.show()

cluster_summary = df.groupby("Cluster").mean(numeric_only=True)
print(cluster_summary)

joblib.dump(kmeams,"customer_segmentation_modelk.pkl")
print("Model saved successfully!")

df.to_csv("Customer_segmentation_result.csv",index=False)
print("Result file saved successfully")

