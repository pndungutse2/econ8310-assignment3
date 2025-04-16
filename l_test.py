import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage
import gzip
import numpy as np
from f_train import CNNFashionMNISTModel, FashionMNISTDataset

# --- Paths ---
test_images = 'data/t10k-images-idx3-ubyte.gz'
test_labels = 'data/t10k-labels-idx1-ubyte.gz'
model_path = 'fashion_mnist_model.pth'

# --- Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Transforms ---
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- Reload Dataset ---
test_dataset = FashionMNISTDataset(test_images, test_labels, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=128)

# --- Load Model ---
model = CNNFashionMNISTModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Evaluate Model ---
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy (Loaded Model): {100 * correct / total:.2f}%")
