# Define CNN model class
import torch
import torch.nn as nn
import torch.optim as optim


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
# print(f"Using device: {device}")

class CNN(nn.Module):
    """
    CNN for Fashion-MNIST classification.
    
    Output shape computation:
      Input: [batch, 1, 28, 28]
      Conv1 (1→6, k=5, s=1, p=0):  (28-5)/1 + 1 = 24  →  [batch, 6, 24, 24]
      ReLU:                         same                   →  [batch, 6, 24, 24]
      MaxPool (k=2, s=2):           24/2 = 12              →  [batch, 6, 12, 12]
      Conv2 (6→16, k=5, s=1, p=0): (12-5)/1 + 1 = 8       →  [batch, 16, 8, 8]
      ReLU:                         same                   →  [batch, 16, 8, 8]
      MaxPool (k=2, s=2):           8/2 = 4                →  [batch, 16, 4, 4]
      Flatten:                      16 * 4 * 4 = 256       →  [batch, 256]
      FC1 (256→120):                                       →  [batch, 120]
      ReLU:                                                 →  [batch, 120]
      FC2 (120→84):                                        →  [batch, 84]
      ReLU:                                                 →  [batch, 84]
      FC3 (84→10):                                         →  [batch, 10]
    """
    def __init__(self):
        super(CNN, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, 5),     # [1, 28, 28] → [6, 24, 24]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # [6, 24, 24] → [6, 12, 12]
            
            nn.Conv2d(6, 16, 5),    # [6, 12, 12] → [16, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(2, 2)      # [16, 8, 8]  → [16, 4, 4]
        )
        
        self.flatten = nn.Flatten() 
        
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),   # 256 → 120
            nn.ReLU(),
            nn.Linear(120, 84),            # 120 → 84
            nn.ReLU(),
            nn.Linear(84, 10)              # 84 → 10
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits

model = CNN().to(device)



def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


def train_cnn(model, train_loader, test_loader, epochs=10, lr=0.01, momentum=0.9, device='cpu'):
    """Train CNN and return training history."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
    print("-" * 56)
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"{epoch+1:<8} {train_loss:<12.4f} {train_acc:<12.4f} {val_loss:<12.4f} {val_acc:<12.4f}")
    
    return history