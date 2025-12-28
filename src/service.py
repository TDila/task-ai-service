import joblib

classifier = joblib.load("model/classifier.joblib")
regressor = joblib.load("model/regressor.joblib")

def predict_task(title: str, description: str):
    text = title + " " + description

    priority = classifier.predict([text])[0]
    time = regressor.predict([text])[0]

    return {
        "priority": priority,
        "estimated_time": round(float(time), 2)
    }