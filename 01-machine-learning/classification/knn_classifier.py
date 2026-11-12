import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "files/datasets" / "telecom_churn_clean.csv"

churn_df = pd.read_csv(CSV_PATH)

X = churn_df[["total_day_charge", "total_eve_charge"]].values
y = churn_df["churn"].values

print(X.shape, y.shape)


knn = KNeighborsClassifier(n_neighbors=15)

knn.fit(X, y)

x_new = np.array([
    [56.8, 17.5],
    [24.4, 24.1],
    [50.1, 10.9]
])

y_predict = knn.predict(x_new)
print("Predictions: {}".format(y_predict))