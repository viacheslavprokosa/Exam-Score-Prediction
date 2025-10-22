from fastapi import FastAPI
from pydantic import BaseModel
from config import MODEL_PATH as model_path

app = FastAPI()

class Student(BaseModel):
    hours_studied: float
    sleep_hours: float
    attendance_percent: float
    previous_scores: float

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(student:Student):
    from joblib import load
    import pandas as pd

    # Load the trained model
    model = load(model_path)

    # Create a DataFrame for the input features
    student_features = pd.DataFrame(
                [[student.hours_studied, student.sleep_hours, student.attendance_percent, student.previous_scores]],
                columns=['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores']
            )
    # Predict the exam score
    predicted_score = model.predict(student_features)
    return {"predicted_score": float(predicted_score[0])}