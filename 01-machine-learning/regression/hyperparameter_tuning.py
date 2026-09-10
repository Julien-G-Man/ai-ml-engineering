"""
Hyperparameters are parameters we specify before fitting a model.
just like alpha and n_neighbors in ridge/lasso regression and KNN
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "files" / "diabetes_clean.csv"

diabetes_df = pd.read_csv(CSV_PATH)

X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values

X_train, X_test, y_train, y_test = train_test_split(
    X ,y, test_size=0.3,                             # 70% training, 30% test
    random_state=21,
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {
    "alpha": np.linspace(0.0001, 1, 10),
    "solver": ["sag", "lsqr"]
}

ridge = Ridge()
ridge_cv = RandomizedSearchCV(ridge, param_grid, cv=kf, n_iter=2)

ridge_cv.fit(X_train, y_train)
print("Tuned Ridge Regression Parameters: {}".format(ridge_cv.best_params_))
print("Tuned Ridge Regression Best Accuracy Score: {}".format(ridge_cv.best_score_))

test_score = ridge_cv.score(X_test, y_test)
print("Test Score: ", test_score)