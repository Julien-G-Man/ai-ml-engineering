"""
Logistic regression for binary classification
  - It's used for classification problems
  - It creates a linear decision boundary  
  - And outputs probabilities
  - If p > 0.5:
       data is labeled 1
  - If p < 0.5:
       data is labeled 0
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report

BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "files" / "diabetes_clean.csv"

diabetes_df = pd.read_csv(CSV_PATH)

X = diabetes_df.drop("diabetes", axis=1).values
y = diabetes_df["diabetes"].values

X_train, X_test, y_train, y_test = train_test_split(
    X ,y, test_size=0.3,                             # 70% training, 30% test
    random_state=42,
    stratify=y
)
    
    
def main():
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_pred_probs = logreg.predict_proba(X_test)[:, 1]
    print("Predictions: ", y_pred_probs[0])
    print("\nROC AUC Scores: ", roc_auc_score(y_test, y_pred_probs))
    print("\nConfusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report: \n", classification_report(y_test, y_pred))
    display_roc_curve(y_test, y_pred_probs)
    
    
def display_roc_curve(y_test, y_pred_probs):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Logistic Regression ROC Curve")
    plt.show()
    
    
if __name__== "__main__":
    main()
    

"""
Note:
Despite being called logistic regression, it is used for classification,
not ordinary regression.

It is called "regression" because the model first learns a linear equation,
similar to linear regression:

    z = w1*x1 + w2*x2 + w3*x3 + b

This linear output, z, is called a logit. Instead of using z directly as the
final prediction, logistic regression passes it through the sigmoid/logistic
function, which converts it into a probability between 0 and 1:

    probability = 1 / (1 + e^-z)

For binary classification:

    if probability >= 0.5:
        predict class 1
    else:
        predict class 0

So the model is regression-like internally, but the final output is used for
classification.

Linear regression predicts continuous values.
Example: predicting glucose level or house price.

Logistic regression predicts class probabilities.
Example: predicting churn / no churn, spam / not spam, disease / no disease.
"""