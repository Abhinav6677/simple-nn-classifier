import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import torch

def plot_accuracy(train_accuracies, test_accuracies):
    """
    Plot training and test accuracy over epochs
    
    Args:
        train_accuracies (list): List of training accuracies for each epoch
        test_accuracies (list): List of test accuracies for each epoch
    """
    # Plotting accuracy vs. epoch
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(test_accuracies, label='Test Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy vs. Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("accuracy_plot.png", dpi=300, bbox_inches='tight')  # saves the plot to file
    plt.show()

