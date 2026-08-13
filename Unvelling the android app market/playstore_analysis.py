import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import plotly.express as px
from textblob import TextBlob
import warnings

warnings.filterwarnings("ignore")

apps = pd.read_csv("googleplaystore.csv")
reviews = pd.read_csv("googleplaystore_user_reviews.csv")
print("Apps Dataset")
print(apps.head())
print(apps.info())
print("\nReviews Dataset")
print(reviews.head())
print(reviews.info())
print(apps.isnull().sum())
print(reviews.isnull().sum())

print("Apps Duplicate: ")
print(apps.duplicated().sum())
print("Reviews Duplicate: ")
print(reviews.duplicated().sum())
apps.drop_duplicates(inplace=True)
reviews.drop_duplicates(inplace=True)
print("Duplicate value after removing: ")
print(apps.duplicated().sum())
print(reviews.duplicated().sum())

plt.figure(figsize=(12,6))
category_counts = apps['Category'].value_counts().head(10)
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.title("Top 10 App Categories")
plt.xlabel("Category")
plt.ylabel("Number of Apps")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(apps['Rating'], bins=20, kde=True)
plt.title("Distribution of App Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

avg_rating = apps.groupby('Category')['Rating'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12,6))
sns.barplot(x=avg_rating.index, y=avg_rating.values)
plt.title("Top 10 Categories by Average Rating")
plt.xlabel("Category")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.show()

apps["Installs"] = apps["Installs"].str.replace(",", "", regex=False)
apps["Installs"] = apps["Installs"].str.replace("+", "", regex=False)
apps["Installs"] = pd.to_numeric(apps["Installs"], errors="coerce")

apps["Size"] = apps["Size"].str.replace("M", "", regex=False)
apps["Size"] = apps["Size"].str.replace("k", "", regex=False)
apps["Size"] = apps["Size"].replace("Varies with device", None)
apps["Size"] = pd.to_numeric(apps["Size"], errors="coerce")

plt.figure(figsize=(10,6))
plt.scatter(apps["Size"], apps["Installs"], alpha=0.5)
plt.title("App Size vs Installs")
plt.xlabel("Size (MB)")
plt.ylabel("Installs")
plt.show()

plt.figure(figsize=(6,6))
apps["Type"].value_counts().plot(kind="pie",autopct="%1.1f%%",startangle=90)
plt.title("Free vs Paid Apps")
plt.ylabel("")
plt.show()

paid_apps = apps[apps["Type"] == "Paid"]
plt.figure(figsize=(8,5))
plt.hist(paid_apps["Price"], bins=20)
plt.title("Price Distribution of Paid Apps")
plt.xlabel("Price")
plt.ylabel("Number of Apps")
plt.show()

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
def get_sentiment(review):
    score = analyzer.polarity_scores(str(review))
    if score['compound'] >= 0.05:
        return "Positive"
    elif score['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"
reviews["Sentiment"] = reviews["Translated_Review"].apply(get_sentiment)
print(reviews[["Translated_Review", "Sentiment"]].head())

plt.figure(figsize=(6,5))
sns.countplot(data=reviews, x="Sentiment")
plt.title("Sentiment Distribution")
plt.show()

merged = reviews.merge(apps[["App", "Category"]],on="App",how="left")
print(merged.head())
sentiment_category = merged.groupby(["Category", "Sentiment"]).size().unstack(fill_value=0)
print(sentiment_category.head())

sentiment_category.plot(kind="bar",stacked=True,figsize=(12,6))
plt.title("Sentiment by Category")
plt.xlabel("Category")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

category_count = apps["Category"].value_counts().reset_index()
category_count.columns = ["Category", "Count"]
fig = px.bar(category_count,x="Category",y="Count",title="Interactive Category Distribution")
fig.show()

merged.to_csv("merged_googleplaystore_reviews.csv", index=False)
print("Merged dataset saved successfully!")

apps.to_csv("cleaned_googleplaystore.csv", index=False)
reviews.to_csv("cleaned_googleplaystore_reviews.csv", index=False)
print("Cleaned datasets saved successfully!")
