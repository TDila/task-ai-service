from fastapi import FastAPI
from pydantic import BaseModel
from src.service import predict_task

app = FastAPI(title="Task AI Service")

class TaskRequest(BaseModel):
    title: str
    description: str

class TaskResponse(BaseModel):
    priority: str
    estimated_time: float

@app.post("/predict", response_model=TaskResponse)
def predict(request: TaskRequest):
    return predict_task(request.title, request.description)