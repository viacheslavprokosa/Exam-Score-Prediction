import mlcroissant as mlc
import pandas as pd


def load_dataset() -> pd.DataFrame:
    from config import DATASET_PATH as dataset_path

    # Importing and fetching dataset from ML Croissant
    croissant_dataset = mlc.Dataset(dataset_path)
    # Check what record sets are in the dataset
    record_sets = croissant_dataset.metadata.record_sets
    # Fetch the records and put them in a DataFrame
    df = pd.DataFrame(croissant_dataset.records(record_set=record_sets[0].uuid))
    # Renaming columns for easier access
    df = df.rename(
        columns={
            "student_exam_scores.csv/student_id": "student_id",
            "student_exam_scores.csv/hours_studied": "hours_studied",
            "student_exam_scores.csv/sleep_hours": "sleep_hours",
            "student_exam_scores.csv/attendance_percent": "attendance_percent",
            "student_exam_scores.csv/previous_scores": "previous_scores",
            "student_exam_scores.csv/exam_score": "exam_score",
        }
    )

    return df


def train_model(df: pd.DataFrame) -> pd.DataFrame:
    from config import MODEL_PATH as model_path
    from joblib import dump
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    X = df[["hours_studied", "sleep_hours", "attendance_percent", "previous_scores"]]
    y = df["exam_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    dump(model, model_path)
    # Add predictions to the DataFrame
    df["predicted_score"] = model.predict(X)
    df["prediction_match"] = [
        "Close" if abs(a - b) < 5 else "Far"
        for a, b in zip(df["exam_score"], df["predicted_score"])
    ]
    return df


def make_pairplot(df: pd.DataFrame):
    import seaborn as sns
    import matplotlib.pyplot as plt

    cols = [
        "hours_studied",
        "sleep_hours",
        "attendance_percent",
        "previous_scores",
        "exam_score",
    ]
    sns.pairplot(
        df[cols + ["prediction_match"]],
        kind="scatter",
        diag_kind="kde",
        hue="prediction_match",
        palette={"Close": "green", "Far": "red"},
    )

    plt.suptitle(
        "Pairplot showing relationships between input features and the exam score, comparing actual vs predicted values",
        y=1.02,
    )
    plt.show()
