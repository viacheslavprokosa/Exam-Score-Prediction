🎯 Exam Score Prediction API

This API predicts a student’s exam score based on study habits and performance metrics.

🚀 Endpoint

POST /predict
URL: http://localhost:8000/predict

📦 Request Format

Content-Type: application/json

Example Request
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"hours_studied": 5.5, "sleep_hours": 7.0, "attendance_percent": 92.5, "previous_scores": 78.0}'

🧾 Request Parameters
Parameter	Type	Description
hours_studied	float	Number of hours the student studied.
sleep_hours	float	Average sleep hours per day.
attendance_percent	float	Attendance percentage.
previous_scores	float	Previous exam scores.
🧠 Response Example
{
  "predicted_score": 85.7
}

⚠️ Common Error Example

If you send an empty or invalid body:

{
  "detail": [
    {
      "type": "missing",
      "loc": ["body"],
      "msg": "Field required",
      "input": null
    }
  ]
}
