import torch
import torch.nn as nn
import torch.optim as optim


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
# print(f"Using device: {device}")

class DNN(nn.Module):
    """
    Fully-connected neural network from Project 1.
    Architecture: 784 → 512 → 512 → 512 → 10 (with ReLU)
    """
    def __init__(self):
        super(DNN, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

# ---- Count parameters ----
from cnn import CNN

dnn_model = DNN().to(device)
dnn_params = sum(p.numel() for p in dnn_model.parameters())
cnn_params = sum(p.numel() for p in CNN().to(device).parameters())

print(f"DNN parameters: {dnn_params:,}")
print(f"CNN parameters: {cnn_params:,}")
print(f"DNN has {dnn_params/cnn_params:.1f}x more parameters than CNN")

# ---- Train DNN on same data ----
print("\n" + "="*60)
print("Training DNN on Fashion-MNIST (same data as CNN)...")
print("="*60)

dnn_model = DNN().to(device)