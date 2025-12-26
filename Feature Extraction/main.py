""" Script to load icon features, train ML models (SVM and MLP),
and evaluate performances"""

import sys
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_path>")
        sys.exit(1)

    json_path = Path(sys.argv[1])

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} not found. Run the extractor first.")
        exit(1)

    # flatten features and labels
    ml_manager = IconMLManager()
    X, y_raw = ml_manager.features_to_vectors(raw_data)

    le = LabelEncoder()  # Encode labels
    y = le.fit_transform(y_raw)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features per sample.")

    # Split data (70% training, 15% val, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Normalize
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

    test_pred_mlp = mlp_model.predict(X_test_scaled)
    print(f"MLP Test Accuracy: {accuracy_score(y_test, test_pred_mlp):.4f}")

    # Classification Report
    print("\n --- MLP Detailed Report ----")
    print(classification_report(y_test, test_pred_mlp,
                                target_names=le.classes_))
