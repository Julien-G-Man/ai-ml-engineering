import random
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """Reduce the activation to a number between 0 and 1"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def ReLU(a: np.array) -> bool:
    """Activation is either active or inactive"""
    return max(0, 1)

def softmax(z):
    """subtract max for numerical stability; doesn't change result"""
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_scores = np.exp(z_shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs

def cross_entropy_loss(y_true, y_pred):
    """\nCross entropy loss function, loss C = -sum(y * log(a))"""
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
    log_likelihood = y_true * np.log(y_pred_clipped)
    loss = -np.mean(np.sum(log_likelihood, axis=1))
    return loss

class SoftmaxNet:
    def __init__(self, in_size, hidden_size, out_size):
        self.W1 = np.random.randn(in_size, hidden_size) * np.sqrt(1 / in_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, out_size) * np.sqrt(1 / hidden_size)
        self.b2 = np.zeros((1, out_size))
        self.loss_history = []
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(self.z2)
        return self.a2
    
    
    def backward(self, X, y, output, lr=0.1):
        """\nBackpropagation using gradient descent
        X is the training data
        y is the desired output
        output is an activation (neuron) from the previous layer
        lr  = learning rate
        a.T = transpose of array a
        m = number of training examples
        
        backprop chain rule: dC/dW = dz/dW * da/dz * dC/da
       
        dC/da2 = 2 * (a2 - y) / m -> we drop the 2, absorbed into lr
        dC_da2 = (output - y) # shape (m, 1)
        
        da2/dz2 = sigmoid deriv(z2)
        dz2 = dC/dz2 = dC/da2 *  da2/dz2
        
        dz2/dW2 = a1
        dC/dW2  = a1.T * dC/dz2 / m
        dz2/db2 = 1
        
        dz2/da1 = W2
        da1 = dC/da1 = dC/dz2 * dz2/da1 # shape (m, hidden)
        
        da1/dz1 = sigmoid'(z1)
        dC/dz1 = dC/da1 * da1/dz1
        
        dz1/dW1 = X
        dC/DW1 = X.T *  dc/dz1 / m
        """
        m = X.shape[0]
        
        # no sigmoid derivative, dC/dz2 = a2 - y when softmax + cross entropy
        dz2 = (output - y) / m
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer
        da1 = np.dot(dz2, self.W2.T)             # how much hidden layer contributed to error
        dz1 = da1 * sigmoid_derivative(self.z1)  # da1/dz1
        dW1 = np.dot(X.T, dz1) / m               # dz1/dW1 = X
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Gradient descent: nudge against gradient
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        
        
    def train(self, X, y, epochs=2000, lr=0.1):
        """ Train network to find the right weights and biases to get the desired output.
            The cost/loss is a measure of how lousy the network is and 
            how badly the system should feel"""
        for epoch in range(epochs):
            output = self.forward(X)           # trash output
            loss = cross_entropy_loss(y, output)
            self.loss_history.append(loss)
            self.backward(X, y, output, lr)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        
    def predict(self, X):
        """Predict output based on trained parameters"""
        output = self.forward(X)
        return (output > 0.5).astype(int)


def plot_loss_curve(loss_history):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.show()

def _class(y: int) -> str:
    return "excercise" if y == 1 else "no excercise"


def main():
    # Sample dataset
    # 3 classes
    # 0=no exercise, 1=exercise, 2=sleep
    X = np.array([
        [7, 2, 0.3],
        [5, 1, 0.2],
        [8, 3, 0.8],
        [4, 2, 0.4],
        [6, 4, 0.7],
        [3, 1, 0.1],
    ])

    # one-hot encoded labels for softmax
    y = np.array([
        [0, 1, 0], # exercise
        [1, 0, 0], # no exercise
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1], # sleep
    ])

    nn = SoftmaxNet(in_size=3, hidden_size=4, out_size=3)
    nn.train(X, y, epochs=1000, lr=0.5)

    person_features  = np.array([[7, 2, 0.3]])
    prediction_prob  = nn.forward(person_features)
    prediction_class = nn.predict(person_features)
    average_loss     = sum(nn.loss_history) / len(nn.loss_history)

    print(f"\nPredicted probability of exercising: {prediction_prob}")
    #print(f"Predicted class: {prediction_class} - ({_class(prediction_class)})")
    print(f"\nAverage loss/cost: {average_loss}")

    plot_loss_curve(nn.loss_history)


if __name__ == "__main__":
    main()
