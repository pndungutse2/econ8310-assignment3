import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gzip
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import ToPILImage
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Custom Dataset ---
class FashionMNISTDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images = self._load_images(images_path)
        self.labels = self._load_labels(labels_path)
        self.transform = transform

    def _load_images(self, path):
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28).astype(np.float32) / 255.0

    def _load_labels(self, path):
        with gzip.open(path, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = ToPILImage()(image)
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Improved CNN Model ---
class CNNFashionMNISTModel(nn.Module):
    def __init__(self):
        super(CNNFashionMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28 -> 14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14 -> 7
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 7 -> 3
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# --- Save Model ---
def save_model(model, path):
    torch.save(model.state_dict(), path)

# --- Paths ---
train_images = 'data/train-images-idx3-ubyte.gz'
train_labels = 'data/train-labels-idx1-ubyte.gz'
test_images = 'data/t10k-images-idx3-ubyte.gz'
test_labels = 'data/t10k-labels-idx1-ubyte.gz'

# --- Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Transforms ---
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(28, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- Datasets and Loaders ---
train_dataset = FashionMNISTDataset(train_images, train_labels, transform=train_transform)
test_dataset = FashionMNISTDataset(test_images, test_labels, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

# --- Model, Loss, Optimizer ---
model = CNNFashionMNISTModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# --- Train Loop ---
for epoch in range(20):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1} - Loss: {running_loss/len(train_loader):.4f}")

# --- Save Final Model ---
save_model(model, 'fashion_mnist_model.pth')

# --- Evaluation and Confusion Matrix ---
model.eval()
correct = total = 0
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"\nTest Accuracy: {100 * correct / total:.2f}%")

# --- Plot Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("FashionMNIST Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
