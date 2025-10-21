import mlcroissant as mlc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns   # For data visualization
import matplotlib.pyplot as plt  # For plotting
from joblib import dump # For saving the model


#1 Start of importing and fetching dataset
# Fetch the Croissant JSON-LD
croissant_dataset = mlc.Dataset('https://www.kaggle.com/datasets/grandmaster07/student-exam-score-dataset-analysis/croissant/download')

# Check what record sets are in the dataset
record_sets = croissant_dataset.metadata.record_sets
# Fetch the records and put them in a DataFrame
df = pd.DataFrame(croissant_dataset.records(record_set=record_sets[0].uuid))
df.head()
#End of importing and fetching dataset
# Renaming columns for easier access
df = df.rename(columns={
    'student_exam_scores.csv/student_id': 'student_id',
    'student_exam_scores.csv/hours_studied': 'hours_studied',
    'student_exam_scores.csv/sleep_hours': 'sleep_hours',
    'student_exam_scores.csv/attendance_percent': 'attendance_percent',
    'student_exam_scores.csv/previous_scores': 'previous_scores',
    'student_exam_scores.csv/exam_score': 'exam_score'
})
# Define features and target variable
X = df[['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores']]
# Target variable
y = df['exam_score']

# 2 Start Machine Learning Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
dump(model, 'student_exam_score_model.joblib')
# End Machine Learning Model Training

# 3 Add prdictions to the DataFrame
df['predicted_score'] = model.predict(X)

# 4 Make pairplot with hue
df['prediction_match'] = ['Close' if abs(a-b)<5 else 'Far' 
                          for a,b in zip(df['exam_score'], df['predicted_score'])]
cols = ['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores', 'exam_score']
sns.pairplot(df[cols + ['prediction_match']], 
             kind='scatter', 
             diag_kind='kde', 
             hue='prediction_match',
             palette={'Close':'green','Far':'red'})

plt.suptitle("Pairplot showing relationships between input features and the exam score, comparing actual vs predicted values", y=1.02)
plt.show()