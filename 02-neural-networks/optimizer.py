import numpy as np
from autograd import Tensor

class Adam:
    def __init__(self, params: list[Tensor], lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.eps = lr, eps
        self.b1, self.b2  = b1, b2
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * p.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * p.grad**2
            m_hat   = self.m[i] / (1 - self.b1 ** self.t)
            v_hat   = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                
    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)