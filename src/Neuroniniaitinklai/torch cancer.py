import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


# Duomenų paruošimas
data = load_breast_cancer()
X = data.data
y = data.target

encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Paverčiame į tensorius
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Duomenų rinkiniai ir DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Modelio struktūra
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fn = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fn(x)

# Inicijuojame modelį, nuostolių funkciją ir optimizatorių

model = SimpleModel(input_size=X_train_tensor.shape[1], output_size=y_train_tensor.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Inicijuojame sąrašus mokymo ir validavimo rezultatams
train_losses = []
val_losses = []
val_accuracies = []


# Modelio apmokymas su rezultatų kaupimu
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, torch.argmax(y_batch, axis=1))
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    train_losses.append(epoch_train_loss / len(train_loader))

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            val_output = model(X_val)
            val_loss += criterion(val_output, torch.argmax(y_val, axis=1)).item()
            _, val_preds = torch.max(val_output, 1)
            val_correct += (val_preds == torch.argmax(y_val, axis=1)).sum().item()
            val_total += y_val.size(0)
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_correct / val_total)

# Braižome nuostolių funkcijas
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss during Training')
plt.legend()

# Braižome tikslumo dinamiką
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy during Training')
plt.legend()

plt.show()

# Modelio testavimas
model.eval()
y_pred_probs = model(X_test_tensor)
y_pred = torch.argmax(y_pred_probs, axis=1).numpy()
y_test_labels = torch.argmax(y_test_tensor, axis=1).numpy()

# Tikslumo spausdinimas
acc = accuracy_score(y_test_labels, y_pred)
print('Test Accuracy:', acc)

# Painiavos matrica
cm = confusion_matrix(y_test_labels, y_pred)
print('Confusion Matrix:\n', cm)

# Painiavos matricos vizualizacija
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Modelio įvertinimas
model.eval()
y_pred_probs = model(X_test_tensor)
y_pred = torch.argmax(y_pred_probs, axis=1).numpy()
y_test_labels = torch.argmax(y_test_tensor, axis=1).numpy()

acc = accuracy_score(y_test_labels, y_pred)
print('Test Accuracy:', acc)
