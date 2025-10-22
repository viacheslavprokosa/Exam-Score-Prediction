import pytest
import pandas as pd
from src.model import train_model

@pytest.fixture
def dummy_df():
    return pd.DataFrame({
        "student_id": [1,2,3],
        "hours_studied": [2,5,3],
        "sleep_hours": [6,7,5],
        "attendance_percent": [80,90,85],
        "previous_scores": [70,75,65],
        "exam_score": [75,88,70],
    })

def test_train_model(dummy_df):
    df_with_pred = train_model(dummy_df)
    assert "predicted_score" in df_with_pred.columns
    assert "prediction_match" in df_with_pred.columns
    assert df_with_pred["predicted_score"].dtype == float
