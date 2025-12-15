import torch
import numpy as np
from sklearn.metrics import classification_report

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def test(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=class_names, labels=np.arange(len(class_names))))

    return all_labels, all_preds

def validate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * images.size(0)
            correct_preds += torch.sum(preds == labels.data)
            total_preds += labels.size(0)
        
    epoch_loss = running_loss / total_preds
    epoch_acc = correct_preds.double() / total_preds

    return epoch_loss, epoch_acc