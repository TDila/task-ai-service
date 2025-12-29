# Task AI Service

A Python-based machine learning microservice that predicts task priority and estimated completion time based on task title and description.

## Tech Stack
- Python
- FastAPI
- scikit-learn
- TF-IDF + RandomForest
- joblib

## Endpoints

### POST /predict

**Request**
```json
{
  "title": "Prepare payroll",
  "description": "Monthly payroll processing"
}
```

**Response**
```json
{
  "priority": "HIGH",
  "estimated_time": 6.5
}
```
## How It Works
1. Text input is vectorized using TF-IDF
2. RandomForest classifier predicts priority
3. RandomForest regressor predicts time
4. Models are loaded from serialized joblib files

## Run locally
- pip install -r requirements.txt
- python src/train.py
- uvicorn src.api:app --reload