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

def evaluate_model_performance(model, X_train, X_test, y_train, y_test, train_accuracies, test_accuracies):
    """
    Comprehensive model evaluation with detailed metrics
    
    Args:
        model: Trained PyTorch model
        X_train, X_test, y_train, y_test: Data splits
        train_accuracies, test_accuracies: Lists of accuracies over epochs
    """
    print("=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Get predictions
        train_preds = torch.argmax(model(X_train), dim=1)
        test_preds = torch.argmax(model(X_test), dim=1)
        
        # Convert to numpy for sklearn metrics
        train_preds_np = train_preds.numpy()
        test_preds_np = test_preds.numpy()
        y_train_np = y_train.numpy()
        y_test_np = y_test.numpy()
    
    # Final accuracy scores
    final_train_acc = accuracy_score(y_train_np, train_preds_np)
    final_test_acc = accuracy_score(y_test_np, test_preds_np)
    
    print(f"\n📊 FINAL ACCURACY SCORES:")
    print(f"   Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"   Test Accuracy:     {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")
    
    # Learning progress analysis
    print(f"\n📈 LEARNING PROGRESS ANALYSIS:")
    print(f"   Starting Train Acc: {train_accuracies[0]:.4f}")
    print(f"   Final Train Acc:    {train_accuracies[-1]:.4f}")
    print(f"   Improvement:        {train_accuracies[-1] - train_accuracies[0]:.4f}")
    
    print(f"   Starting Test Acc:  {test_accuracies[0]:.4f}")
    print(f"   Final Test Acc:     {test_accuracies[-1]:.4f}")
    print(f"   Improvement:        {test_accuracies[-1] - test_accuracies[0]:.4f}")
    
    # Overfitting analysis
    overfitting = final_train_acc - final_test_acc
    print(f"\n⚠️  OVERFITTING ANALYSIS:")
    print(f"   Train-Test Gap:     {overfitting:.4f}")
    if overfitting > 0.1:
        print("   ⚠️  WARNING: Potential overfitting detected!")
    elif overfitting < -0.05:
        print("   ⚠️  WARNING: Model might be underfitting!")
    else:
        print("   ✅ Good generalization (low overfitting)")
    
    # Performance assessment
    print(f"\n🎯 PERFORMANCE ASSESSMENT:")
    if final_test_acc > 0.9:
        print("   🏆 EXCELLENT: Model performs very well!")
    elif final_test_acc > 0.8:
        print("   ✅ GOOD: Model performs well")
    elif final_test_acc > 0.6:
        print("   ⚠️  FAIR: Model needs improvement")
    elif final_test_acc > 0.4:
        print("   ❌ POOR: Model needs significant improvement")
    else:
        print("   💀 VERY POOR: Model is essentially random")
    
    # Detailed classification report
    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_test_np, test_preds_np, 
                              target_names=['Setosa', 'Versicolor', 'Virginica']))
    
    # Confusion matrix
    print(f"\n🔍 CONFUSION MATRIX:")
    cm = confusion_matrix(y_test_np, test_preds_np)
    print("Predicted →")
    print("Actual ↓")
    print("           Setosa  Versicolor  Virginica")
    print(f"Setosa      {cm[0][0]:>7}  {cm[0][1]:>10}  {cm[0][2]:>9}")
    print(f"Versicolor  {cm[1][0]:>7}  {cm[1][1]:>10}  {cm[1][2]:>9}")
    print(f"Virginica   {cm[2][0]:>7}  {cm[2][1]:>10}  {cm[2][2]:>9}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if final_test_acc < 0.4:
        print("   • Increase learning rate")
        print("   • Add more layers or neurons")
        print("   • Train for more epochs")
        print("   • Check data preprocessing")
        print("   • Consider different optimizer")
    elif final_test_acc < 0.7:
        print("   • Fine-tune hyperparameters")
        print("   • Try different architectures")
        print("   • Add regularization")
    else:
        print("   • Model is performing well!")
        print("   • Consider ensemble methods for even better performance")
    
    print("=" * 60)

# Example usage (uncomment if running this file directly):
# if __name__ == "__main__":
#     # This would be called from data_training.py with actual data
#     plot_accuracy(train_accuracies, test_accuracies)
