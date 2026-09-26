import numpy as np
from collections import Counter

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        
    def euclidean_distance(self, X1, X2):
        """ Euclidean distance between two point
            d = √(Σ(x₁ - x₂)²)"""
        return np.sqrt(np.sum((X1 - X2) ** 2))
    
    def fit(self, X, y):
        """Store training data (no actual training needed)"""
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        """Predict class labels for all samples in X"""
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x):
        """Predict class labels for a single sample"""
        # calculate distances for all training samples
        distances = [self.euclidean_distance(x, x_train) 
                     for x_train in self.X_train]
        
        # Get indices of K nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # get labels of k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def _predict_single_regression(self, x):
        """ KNN Regression Implementation.
            Returns average instead of voting"""
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_values = [self.y_tran[i] for i in k_indices]
        return np.mean(k_nearest_values) 


def main():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(
        n_samples=100, n_features=2,
        n_redundant=0, random_state=42)
    
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
    
    model = KNearestNeighbors(k=5)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Predictions: {predictions}")
    print(f"Accuracy: {accuracy:.2f}")
    
    

if __name__ == "__main__":
    main()