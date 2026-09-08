"""
y = ax + b

y -> target
x = single feature
a, b -> parameters/coefficients of the model slop, intercept

How do we choose a and b?
  - Define an error function for any given line
  - Choose the line that minimizes the error function
  
Error function = loss function = cost function


The error function in linear regression is called Residual Sum of Squares (RRS)
In Ordinary Least Squares (OLS - this type of linear regression), the aim is to minimise RRS 
LR  preforms OLS under the hood

given:
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5,  0.0, 2.0, 8.0])

The Residual Sum of Squares will be
    rss = np.sum((y_true - y_pred) ** 2)

    resisual = y_true - y_pred -->> error or vertical distance between actual data point and the prodicted line

Linear regression in higher dimensions
    y = a1*x1 + a2*x2 + b
"""
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error


BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "files" / "diabetes_clean.csv"

diabetes_df = pd.read_csv(CSV_PATH)

X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values

X_bmi = X[:, 3]
# print(y.shape, X_bmi.shape)

# convert to 2D array
X_bmi = X_bmi.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)

# R^2 quantifies the target values explained by the features
# values ranhe from 0 to 1

r_squared = reg_all.score(X_test, y_test)

# Mean Squared Error, MSE  = i/n * (rss)
# RMSE = sqrt(MSE)
root_mse = root_mean_squared_error(y_test, y_pred)

print("R^2: {}".format(r_squared))
print("RMSE: {}".format(root_mse))