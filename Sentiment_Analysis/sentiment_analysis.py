import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib

df = pd.read_csv("Dataset.csv",encoding="latin-1",header=None)
df.columns = ["target","id","date","flag","user","text"]
print(df.head())
print(df.shape)
print(df.info())
df = df.sample(n=50000,random_state=42)
df["target"]= df["target"].replace(4,1)
print(df["target"].value_counts())
print(df.shape)
df = df[["target","text"]]
print(df.head())

def clean_text(text):
    text= text.lower()
    text=re.sub(r"http\S+","",text)
    text=re.sub(r"@\W+", "", text)
    text=re.sub(r"[^a-zA-Z\s]", "", text)
    return text
df["clean_text"]=df["text"].apply(clean_text)
print(df[["text","clean_text"]].head())

from sklearn.model_selection import train_test_split
X = df["clean_text"]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train.shape)
print(X_test.shape)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
print(X_train.shape)
print(X_test.shape)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
df.to_csv("sentiment140_sample.csv",index=False)