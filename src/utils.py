import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def plot_curves(results, model_name):
    train_loss = results['train_loss']
    val_loss = results['val_loss']

    train_acc = results['train_acc']
    val_acc = results['val_acc']

    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epochs')
    plt.legend()

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

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen')

    if not os.path.exists("plots"):
        os.makedirs("plots")
    
    save_path = f"plots/{model_name}_conf_matrix.png"
    plt.savefig(save_path)
    plt.close()