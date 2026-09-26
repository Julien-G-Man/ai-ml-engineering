import torch
from pathlib import Path
from torch.optim import Adam
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "pytorch_model_v1.pth"

# dummy data
X = torch.rand((100, 10))
y = torch.randint(0, 2, (100,))

model = Sequential(
    Linear(10, 32),
    ReLU(),
    Linear(32, 2)
)


def main():
    # loss and optimizer
    criterion = CrossEntropyLoss()
    optimizer  = Adam(model.parameters(), lr=0.01)

    # training loop
    for epoch in range(10):
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Training Complete.")
    save_model_state(MODEL_PATH)
    

def save_model_state(model_path):
    torch.save(model.state_dict(), model_path)
    

def load_model_state(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    
if __name__ == "__main__":
    main()