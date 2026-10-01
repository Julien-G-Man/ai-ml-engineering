"""
Basic PyTorch implemention of neural network from basic_neuralnet.py
"""


import torch
import torch.nn as nn
import matplotlib.pyplot as plt


X = torch.tensor([
    [7, 2, 0.3],
    [5, 1, 0.2],
    [8, 3, 0.8],
    [4, 2, 0.4],
    [6, 4, 0.7],
    [3, 1, 0.1],
], dtype=torch.float32)

y = torch.tensor([1, 0, 1, 0, 1, 2])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(3, 4) # in=3, hidden=4
        self.W2 = nn.Linear(4, 3) # hidden=4, out=3
        
        
    def forward(self, x):
        """We return logits, not softmax
        loss does softmax internally"""
        z1 = self.W1(x)
        a1 = torch.sigmoid(z1)
        z2 = self.W2(a1)
        return z2
    
    
def plot_loss_curve(loss_history):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.show()
    
    
model = Net()
loss_fn = nn.CrossEntropyLoss() # softmax + cross entropy loss combined
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
loss_history = []

# train
for epoch in range(1000):
    optimizer.zero_grad()       # clear old dW
    logits = model(X)           # forward -> z2
    loss = loss_fn(logits, y)   # softmax(logits) + -y*log(a)
    loss_history.append(loss)
    loss.backward()
    optimizer.step()            # W -= lr * dW
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        
# predict
with torch.no_grad():
    probs = torch.softmax(model(X[0]), dim=0)
    print(probs)
    plot_loss_curve(loss_history)