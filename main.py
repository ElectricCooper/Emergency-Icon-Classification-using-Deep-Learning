"""Script to run training and evaluation pipeline for multiple models."""
import os

import torch
import torch.nn as nn
import torch.optim as optim

from src import data_setup, model_setup, engine, utils

DATA_PATH = "data/extracted"
MODELS = ["resnet50", "vit_b_16"]
EPOCHS = 20
BATCH_SIZE = 32
LR = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """Run training and evaluation pipeline"""
    print(f"Working on {DEVICE}")

    train_loader, val_loader, test_loader, class_names = (
        data_setup.create_dataloaders(DATA_PATH, BATCH_SIZE))

    if not os.path.exists("models"):
        os.makedirs("models")

    for model_name in MODELS:
        print(f"\n{'-'*30}")
        print(f"NOW TRAINING {model_name}")
        print(f"\n{'-'*30}")

        model = model_setup.get_model(model_name, len(class_names), DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        results = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        for epoch in range(EPOCHS):
            train_loss, train_acc = engine.train(
                            model,
                            train_loader,
                            criterion,
                            optimizer,
                            DEVICE
                        )
            
            val_loss, val_acc = engine.validate(
                                    model,
                                    val_loader,
                                    criterion,
                                    DEVICE
                                )

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc.item() if isinstance(val_acc, torch.Tensor) else val_acc)

            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        print(f"\n--- {model_name} TEST RESULTS ---")

        y_true, y_pred = engine.test(model, test_loader, DEVICE, class_names)

        utils.plot_curves(results, model_name)

        utils.plot_confusion_matrix(y_true, y_pred, class_names, model_name)

        torch.save(model.state_dict(), f"models/{model_name}_final.pth")
        print("Model Saved")

        del model
        # Emptying cache as we are training models back to back
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
