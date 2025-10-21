from joblib import load
import pandas as pd

# Load the trained model
model = load('student_exam_score_model.joblib')

# Get student features from user input
hours_studied = float(input("Hours studiet: "))
sleep_hours = float(input("Sleep hours: "))
attendance_percent = float(input("Attendance percent (%): "))
previous_scores = float(input("Previous scores: "))

# Create a DataFrame for the input features
student_features = pd.DataFrame(
            [[hours_studied, sleep_hours, attendance_percent, previous_scores]],
            columns=['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores']
        )
# Predict the exam score
predicted_score = model.predict(student_features)
print(f"Predicted score: {predicted_score[0]:.2f}")
