import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
import joblib
import os

#load data
df = pd.read_csv("data/tasks.csv")

x = df["title"] + " " + df["description"]
y_priority = df["priority"]
y_time = df["hours"]

priority_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("reg", LogisticRegression(max_iter=1000))
])

priority_model.fit(x, y_priority)

time_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("reg", LinearRegression())
])

time_model.fit(x, y_time)

#save models
os.makedirs("model", exist_ok=True)
joblib.dump(priority_model, "model/classifier.joblib")
joblib.dump(time_model, "model/regressor.joblib")

print("Models trained and saved successfully")