"""
CROSS-VALIDATION MOTIVATION
 - Model performance is dependent on the way we split up the data
 - Not representative of the model's ability to generalize to unseen data
  - Solution: Cross validation
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "files" / "diabetes_clean.csv"

diabetes_df = pd.read_csv(CSV_PATH)

X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values

kf = KFold(n_splits=6, shuffle=True, random_state=42)
reg = LinearRegression()

cv_results = cross_val_score(reg, X, y, cv=kf)
print(cv_results)
print(f"Mean CV score:      {np.mean(cv_results)}")
print(f"Standard Deviation: {np.std(cv_results)}")
print(f"95% confidence interval: {np.quantile(cv_results, [0.025, 0.975])}")