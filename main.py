import torch
import torch.nn as nn
import torch.optim as optim
import os
from src import data_setup, model_setup, engine

DATA_PATH = "data/extracted"
MODELS = ["resnet50", "vit_b_16"]
EPOCHS = 20
BATCH_SIZE = 32
LR = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Working on {DEVICE}")

    train_loader, val_loader, test_loader, class_names = data_setup.create_dataloaders(DATA_PATH, BATCH_SIZE)

    if not os.path.exists("models"):
        os.makedirs("models")

    for model_name in MODELS:
        print(f"\n{'-'*30}")
        print(f"NOW TRAINING {model_name}")
        print(f"\n{'-'*30}")

        model = model_setup.get_model(model_name, len(class_names), DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        for epoch in range(EPOCHS):
            train_loss = engine.train(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc = engine.validate(model, val_loader, criterion, DEVICE)

            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        print(f"\n--- {model_name} TEST RESULTS ---")

        engine.test(model, test_loader, DEVICE, class_names)

        torch.save(model.state_dict(), f"models/{model_name}_final.pth")
        print("Model Saved")

        del model 
        torch.cuda.empty_cache() #Emptying the cache because we're training the models back to back

if __name__ == "__main__":
    main()