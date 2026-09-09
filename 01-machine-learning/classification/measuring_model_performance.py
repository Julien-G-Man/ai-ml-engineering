"""
The most commonly used metric for classification is Accuracy

Accuracy = correct predictions / total observations

To computer accuracy, we split the data in training set and test set
We then fit/train the classifer on training set, and test performace using test set
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as  plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "files" / "telecom_churn_clean.csv"

churn_df = pd.read_csv(CSV_PATH)

X = churn_df[["total_day_charge", "total_eve_charge"]].values
y = churn_df["churn"].values

X_train, X_test, y_train, y_test = train_test_split(
    X ,y, test_size=0.3,                             # 70% training, 30% test
    random_state=21, stratify=y
)

train_accuracies = {}
test_accuracies = {}
neighbors = np.arange(1, 26)
    
for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor]  = knn.score(X_test, y_test)

def plot_results():
    plt.figure(figsize=(8, 6))
    plt.title("KNN: Varying Number of Neighbors")
    plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
    plt.plot(neighbors, test_accuracies.values(), label="Test Accuracy")
    plt.legend()
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Accuracy")
    plt.show()
    
plot_results()
