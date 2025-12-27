import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def plot_curves(results, model_name):
    """
    Plots training and validation loss/accuracy curves for Deep Learning models.
    """
    train_loss = results['train_loss']
    val_loss = results['val_loss']

    train_acc = results['train_acc']
    val_acc = results['val_acc']

    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(15, 7))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Val Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    if not os.path.exists("plots"):
        os.makedirs("plots")

    save_path = f"plots/{model_name}_curves.png"
    plt.savefig(save_path)
    plt.close()

def plot_mlp_loss_curve(mlp_model, model_name="MLP"):
    """
    Plots the loss curve for Scikit-Learn MLPClassifier.
    """
    if not hasattr(mlp_model, 'loss_curve_'):
        print(f"Warning: {model_name} does not have a loss curve (maybe max_iter was too low or warm_start used).")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(mlp_model.loss_curve_, label='Training Loss')
    plt.title(f'{model_name} - Training Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    if not os.path.exists("plots"):
        os.makedirs("plots")

    save_path = f"plots/{model_name}_loss_curve.png"
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """
    Plots confusion matrix for both DL and ML models.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted')

    if not os.path.exists("plots"):
        os.makedirs("plots")
    
    save_path = f"plots/{model_name}_conf_matrix.png"
    plt.savefig(save_path)
    plt.close()