import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "files" / "diabetes_clean.csv"

diabetes_df = pd.read_csv(CSV_PATH)

X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values

X_bmi = X[:, 3]
# print(y.shape, X_bmi.shape)

# convert to 2D array
X_bmi = X_bmi.reshape(-1, 1)

reg = LinearRegression()
reg.fit(X_bmi, y)
predictions = reg.predict(X_bmi)

    

def plot_results(X_bmi, predictions):
    plt.scatter(X_bmi, predictions)
    plt.ylabel("Blood Glucose (mg/dl)")
    plt.xlabel("Body Mass Index")
    plt.show()
    
if __name__ == "__main__":
    plot_results(X_bmi, predictions)