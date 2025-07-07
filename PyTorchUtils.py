import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from tqdm import tqdm


def train_model(model: nn.Module,train_loader, optimizer: Optimizer, criterion: _Loss, epochs, device: torch.device):
    model.train()
    total_loss = []

    for epoch in range(epochs):

        running_loss = 0.0
        progress_bar = tqdm(train_loader,
                            desc=f"Epoch: {epoch + 1} / {epochs}",
                            unit="batch")

        for i, (images, labels) in enumerate(progress_bar, 1):
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(avg_loss=running_loss / i)
        
        total_loss.append(running_loss)
    
    return total_loss

def evaluate_model(model: nn.Module, test_loader, device):
    model.eval()
    correct = total = 0
    progress_bar = tqdm(test_loader,
                        desc=f"Test Dataset",
                        unit="batch")
    with torch.no_grad():
        for i, (images, labels) in enumerate(progress_bar, 1):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Test Accuracy: {100 * correct / total:.2f}%")



