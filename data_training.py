import torch
import torch.nn as nn
import torch.optim as optim
from data_model import Model
from data_loader import load_and_prepare_data
from sklearn.metrics import accuracy_score
from evaluation import plot_accuracy

X_train, X_test, y_train, y_test = load_and_prepare_data()

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_accuracies = []
test_accuracies = []

# Training loop
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        train_preds = torch.argmax(model(X_train_tensor), dim=1)
        test_preds = torch.argmax(model(X_test_tensor), dim=1)

        train_acc = accuracy_score(y_train_tensor.numpy(), train_preds.numpy())
        test_acc = accuracy_score(y_test_tensor.numpy(), test_preds.numpy())

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    print(f"Epoch {epoch+1}/50 - Loss: {loss.item():.4f} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")

print("\nTraining completed! Plotting results...")
plot_accuracy(train_accuracies, test_accuracies)
