"""Script to run training and evaluation pipeline for multiple models (DL + ML)."""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

from src import data_setup, model_setup, engine, utils

# Constants
DATA_PATH = "data/extracted"
JSON_FEATURES_PATH = "features_dataset.json" 

MODELS = ["simple_cnn", "resnet50", "vit_b_16", "resnet18"]
EPOCHS = 10
BATCH_SIZE = 32
LR = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Working on {DEVICE}')


class IconMLManager:
    """Does data loading and model training for icon features"""

    @staticmethod
    def features_to_vectors(dataset):
        """Flatten JSON structure into a matrix X and a label matrix y"""
        _x = []
        _y = []

        for sample in dataset:
            feature_vector = []

            # process subdivisions
            for sub in sample['subdivisions']:
                feature_vector.extend([
                    sub['perimeter'],
                    sub['area'],
                    sub['compactness'],
                    sub['corners_count'],
                    sub['sharp_corners_count']
                ])
                feature_vector.extend(sub['hu_moments'])
                feature_vector.extend([
                    sub['line_directions']['horizontal'],
                    sub['line_directions']['vertical'],
                    sub['line_directions']['diag1'],
                    sub['line_directions']['diag2']
                ])

            # process global features
            g = sample['global']
            feature_vector.extend([
                g['perimeter'],
                g['area'],
                g['compactness'],
                g['corners_count'],
                g['sharp_corners_count'],
                g['ellipse_count'],
                g['diagonal_length'],
                g['diagonal_angle'],
                g['convex_area']['convex_area'],
                g['convex_area']['solidity'],
                g['avg_centroidal_radius']
            ])
            feature_vector.extend(g['hu_moments'])
            feature_vector.extend([
                g['line_directions']['horizontal'],
                g['line_directions']['vertical'],
                g['line_directions']['diag1'],
                g['line_directions']['diag2']
            ])

            _x.append(feature_vector)
            _y.append(sample['label'])

        return np.array(_x), np.array(_y)


def train_ml(json_path_str):
    """Runs the SVM and MLP training using extracted features"""
    print(f"\n{'='*30}")
    print("STARTING TRADITIONAL ML PIPELINE (SVM & MLP)")
    print(f"{'='*30}")

    json_path = Path(json_path_str)

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} not found. Ensure extracted features exist.")
        return

    ml_manager = IconMLManager()
    X, y_raw = ml_manager.features_to_vectors(raw_data)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features per sample.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # SVM
    print("\n--- Training SVM ---")
    svm_model = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    val_pred_svm = svm_model.predict(X_val_scaled)
    test_pred_svm = svm_model.predict(X_test_scaled)
    print(f"SVM Val Accuracy: {accuracy_score(y_val, val_pred_svm):.4f}")
    print(f"SVM Test Accuracy: {accuracy_score(y_test, test_pred_svm):.4f}")

    utils.plot_confusion_matrix(y_test, test_pred_svm, le.classes_, "SVM")

    # MLP
    print("\n --- Training MLP ---")
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    mlp_model.fit(X_train_scaled, y_train)

    val_pred_mlp = mlp_model.predict(X_val_scaled)
    print(f"MLP Val Accuracy: {accuracy_score(y_val, val_pred_mlp):.4f}")
    
    test_pred_mlp = mlp_model.predict(X_test_scaled)
    print(f"MLP Test Accuracy: {accuracy_score(y_test, test_pred_mlp):.4f}")

    print("\n --- MLP Detailed Report ----")
    print(classification_report(y_test, test_pred_mlp,
                                target_names=le.classes_))
    
    utils.plot_confusion_matrix(y_test, test_pred_mlp, le.classes_, "MLP")
    utils.plot_mlp_loss_curve(mlp_model, "MLP")


def main():
    """Run training and evaluation pipeline"""
    print(f"Working on {DEVICE}")

    # --- TRADITIONAL ML PART ---
    train_ml(JSON_FEATURES_PATH)
    
    # --- DEEP LEARNING PART ---
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
        torch.cuda.empty_cache()



if __name__ == "__main__":
    main()