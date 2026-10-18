import numpy as np

class MLP:
    """Multi-Layer Perceptron"""
    def __init__(self, n_in: int, n_hidden: int, n_out: int):
        self.W1 = np.random.randn(n_in, n_hidden) * np.sqrt(1 / n_in)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_out) * np.sqrt(1 / n_hidden)
        self.b2 = np.zeros((1, n_out))
        self.loss_history: list[float] = []
        
        # adam buffers
        self.mW1, self.vW1 = np.zeros_like((self.W1)), np.zeros_like((self.W1))
        self.mb1, self.vb1 = np.zeros_like((self.b1)), np.zeros_like((self.b1))
        self.mW2, self.vW2 = np.zeros_like((self.W2)), np.zeros_like((self.W2))
        self.mb2, self.vb2 = np.zeros_like((self.b2)), np.zeros_like((self.b2))
        self.t = 0
        
    def forward(self, X):
        self.X = X
        self.z = np.dot(X, self.W1) + self.b1
        self.a = sigmoid(self.z)
        self.logits = np.dot(self.a, self.W2) + self.b2
        self.probs  = softmax(self.logits)
        return self.probs
    
    def backward(self, y, lr=1e-3):
        B = y.shape[0]
        dlogits = self.probs.copy()
        dlogits[np.arange(B), y] -= 1
        dlogits /= B
        
        dW2 = np.dot(self.a.T, dlogits)
        db2 = dlogits.sum(0)
        da  = np.dot(dlogits, self.W2.T)
        dh  = da * self.a * (1 - self.a)
        dW1 = np.dot(self.X.T, dh)
        db1 = dh.sum(0)
        
        self.t += 1
        self._adam(self.W1, dW1, self.mW1, self.vW1, lr=lr)
        self._adam(self.b1, db1, self.mb1, self.vb1, lr=lr)
        self._adam(self.W2, dW2, self.mW2, self.vW2, lr=lr)
        self._adam(self.b2, db2, self.mb2, self.vb2, lr=lr)
        
        return cross_entropy(self.probs, y)
    
    def _adam(self, p, g, m, v, lr=1e-3, b1=0.9, b2=0.999):
        m[:]  = (b1 * m) + (1 - b1) * g 
        v[:]  = (b2 * v) + (1 - b2) * (g * g) 
        m_hat = m / (1 - b1**self.t)
        v_hat = v / (1 - b2**self.t)
        p    -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
            
    def train(self, X, y, epochs: int=5, batch: int=5, lr=1e-3, verbose=True):
        for epoch in range(epochs):
            idx = np.random.permutation(len(X))
            X_shuf, y_shuf = X[idx], y[idx]
            for i in range(0, len(X), batch):
                xb = X_shuf[i:i + batch]
                yb = y_shuf[i:i + batch]
                self.forward(xb)
                loss = self.backward(yb, lr=lr)
                self.loss_history.append(loss)
            if verbose:
                d = 1/5 * epochs if epochs < 10 else 1/10 * epochs
                if epoch % d == 0:
                    print(f"epoch {epoch+1}/{epochs}, loss: {loss:.3f}")
                    


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(logits):
    max_v = np.max(logits, axis=1, keepdims=True)
    e = np.exp(logits - max_v)
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy(probs, y):
    B = y.shape[0]
    n = probs[np.arange(B), y]
    return -np.log(n + 1e-8).mean()

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
    # return np.dot(x, w)

def ReLU(a) -> bool:
    return max(0, 1)