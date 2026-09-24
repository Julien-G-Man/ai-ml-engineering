"""
Sample Implementation of Linear Regression
    
The Mathematics Behind It
i. The Linear Equation
    y = mx + b
    
ii. Cost Function (MSE)
  - measures prediction error
    J = (1/n) Σ(y_pred - y_actual)²

iii. Gradient Descent 
  - Iteratively minimizes cost function by updating parameters
    m = m - α * ∂J/∂m
    b = b - α * ∂J/∂b


Parameter Update Rules
    ∂J/∂m = (2/n) Σx(y_pred - y_actual)
    ∂J/∂b = (2/n) Σ(y_pred - y_actual)


Key Hyperparameters to Tune
  1. Learning Rate (α - alpha) 
    - controls the step size in gradient descent
    - too high: overshooting, instability
    - too low: slow convergence
    - typical range: 0.001 to 0.1

  2. Number of Iterationss
    - how many times to update parameters
    - more iterations: better convergence
    - watch for diminishing returns
    - typical range: 100 to 10,000

"""

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.m = 0 # slop
        self.b = 0 # intercept
        self.cost_history = []
        
    def predict(self, X):
        return self.m * X + self.b
    
    def fit(self, X, y):
        """Train the model using gradient descent"""
        n = len(X)
        
        for i in range(self.iterations):
            y_pred = self.predict(X)
            
            # calculate gradients & update parameters (gradient descent)
            dm = (2/n) * np.sum(X * (y_pred - y))
            db = (2/n) * np.sum(y_pred - y)
            
            self.m -= self.learning_rate * dm
            self.b -= self.learning_rate * db
            
            # Store cost/loss
            cost = mean_squared_error(y_pred, y)
            self.cost_history.append(cost)
            

def mean_squared_error(y_pred, y_actual):
    """cost/loss function"""
    return np.mean((y_pred - y_actual) ** 2)


def plot_loss_curve(cost_history):
    plt.figure(figsize=(8, 5))
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost / Loss")
    plt.title("Loss / Cost Curve")
    plt.grid(True)
    plt.plot(cost_history)
    plt.show()
    

def main():
    # sample data
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([2, 4, 5, 4, 5, 7, 8, 8, 9, 10])
    
    model = LinearRegression(learning_rate=0.01, iterations=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    
    print(f"Predictions:   {predictions}")
    print(f"\nSlope     (m): {model.m:.4f}")
    print(f"Intercept (b): {model.b:.4f}")
    plot_loss_curve(model.cost_history)
    

if __name__ == "__main__":
    main()