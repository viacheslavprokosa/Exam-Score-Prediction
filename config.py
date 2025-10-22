from dotenv import load_dotenv
import os

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "student_exam_score_model.joblib")
DATASET_PATH = os.getenv("DATASET_PATH", "https://www.kaggle.com/datasets/grandmaster07/student-exam-score-dataset-analysis/croissant/download")