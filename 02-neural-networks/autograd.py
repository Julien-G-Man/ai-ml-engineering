"""
Autograd Tensor that recursively applies chain rule through a computational graph
"""
import numpy as np

class Tensor:
    def __init__(self, data, _prev=(), _op=''):
        self.data  = np.array(data) if not isinstance(data, np.ndarray) else data
        self.grad  = np.zeros_like(self.data)
        self._prev = set(_prev)
        self._op   = _op
        self._backward = lambda: None
        
        
    def __matmul__(self, other):
        """Matrix Multiplication"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out   = Tensor(np.dot(self.data, other.data), (self, other), '@')
        
        def _backward():
            self.grad  += out.grad @ other.data.T
            # handle batch matmul
            if other.data.ndim == 2:
                other.grad += self.data.T @ out.grad
            else:
                other.grad += self.data.T
        out._backward = _backward
        return out
        
        
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out   = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad  += out.grad
            other.grad += out.grad.sum(axis=0) if other.grad.shape !=  out.grad.shape else out.grad
        out._backward = _backward
        return out
        
        
    def sigmoid(self):
        s   = 1 / (1 + np.exp(-self.data))
        out = Tensor(s, (self,), 'sigmoid')
        
        def _backward():
            self.grad += (s * (1-s)) * out.grad
        out._backward = _backward
        return out
       
    
    def softmax(self):
        x = self.data - self.data.max(axis=1, keepdims=True)
        e = np.exp(x)
        p = e / e.sum(axis=1, keepdims=True)  
        out = Tensor(p, (self, ), 'softmax')
        
        def _backward():
            self.grad += out.grad
        out._backward = _backward
        out._softmax_data = p
        return out   
            
            
    def backward(self):
        # topological sort
        topo = []
        visited = set()
        def build(v: Tensor):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()
