"""
The confusion matrix is a simple table used to evaluate how well a classification model performs.
It compares what the model predicted against the real, actual values in a dataset.

Four outcomes:
  - True Positive (TP):  The model says yes, and the real answer is yes.
  - True Negative (TN):  The model says no,  and the real answer is no.
  - False Positive (FP): The model says yes, but the real answer is no.  This is a Type I error (a false alarm).
  - False Negative (FN): The model says no,  but the real answer is yes. This is a Type II error (a missed case). 
"""

import pandas as pd
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "files" / "diabetes_clean.csv"

diabetes_df = pd.read_csv(CSV_PATH)

X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values

X_train, X_test, y_train, y_test = train_test_split(
    X ,y, test_size=0.4,                             # 60% training, 40% test
    random_state=42,
)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train , y_train)

y_pred = knn.predict(X_test)
print("The Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

"""
The result looks something like this:
    [[TN  FP]
     [FN  TP]]
"""

print("\nClassification Report:\n", classification_report(y_test, y_pred))