import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "files" / "diabetes_clean.csv"

diabetes_df = pd.read_csv(CSV_PATH)

X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values

X_train, X_test, y_train, y_test = train_test_split(
    X ,y, test_size=0.3,                             # 70% training, 30% test
    random_state=21,
)

scores = {
    "ridge": [],
    "lasso": []
}

def ridge_regression():
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        scores["ridge"].append(ridge.score(X_test, y_test))
    
def lasso_regression():
    for alpha in [0.01, 1.0, 10.0, 20.0, 50.0]:
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)
        lasso_pred = lasso.predict(X_test)
        scores["lasso"].append(lasso.score(X_test, y_test))
       
def lasso_feature_selection():
    names = diabetes_df.drop("glucose", axis=1).columns
    lasso = Lasso(alpha=0.1) 
    lasso_coef = lasso.fit(X, y).coef_
    plt.bar(names, lasso_coef)  
    plt.xticks(rotation=45)
    plt.show()
     
        
def main():
    ridge_regression()
    lasso_regression()
    print(json.dumps(scores, indent=2))
    lasso_feature_selection()



if __name__ == "__main__":
    main()