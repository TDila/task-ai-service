import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import os

#load data
df = pd.read_csv("data/tasks.csv")

x = df["title"] + " " + df["description"]
y_priority = df["priority"]
y_time = df["hours"]

priority_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("reg", RandomForestClassifier(n_estimators=100, random_state=42))
])

priority_model.fit(x, y_priority)

time_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("reg", RandomForestRegressor(n_estimators=100, random_state=42))
])

time_model.fit(x, y_time)

#save models
os.makedirs("model", exist_ok=True)
joblib.dump(priority_model, "model/classifier.joblib")
joblib.dump(time_model, "model/regressor.joblib")

print("Models trained and saved successfully")