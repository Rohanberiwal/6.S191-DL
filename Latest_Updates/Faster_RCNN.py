import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm  # For progress bars

def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, targets in tqdm(data_loader, desc="Training Epoch"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        running_loss += losses.item()
    
    epoch_loss = running_loss / len(data_loader)
    return epoch_loss

def evaluate(model, data_loader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            running_loss += losses.item()
    
    epoch_loss = running_loss / len(data_loader)
    return epoch_loss

def train(model, train_loader, val_loader, num_epochs, device):
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")

        val_loss = evaluate(model, val_loader, device)
        print(f"Validation Loss: {val_loss:.4f}")

# Example usage
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

num_classes = 2  # Background + 1 class (adjust based on your dataset)
model = get_model(num_classes)  # Ensure this function is defined as in previous examples

num_epochs = 10
train(model, train_loader, val_loader, num_epochs, device)
