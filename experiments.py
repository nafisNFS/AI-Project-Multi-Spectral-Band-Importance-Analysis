import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import rasterio
from torch.utils.data import DataLoader, Dataset
import os


# Define CNN Architecture
class CustomCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CustomCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.GELU(),
            nn.MaxPool2d(2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Custom Dataset Loader for EuroSAT All-Bands
class EuroSATDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.filepaths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith(".tif"):
                    self.filepaths.append(os.path.join(class_dir, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float32)  # Shape: (bands, H, W)
        label = self.labels[idx]
        if self.transform:
            img = self.transform(torch.tensor(img))
        return img, label


# Load dataset
data_transform = transforms.Compose([
    transforms.Normalize(mean=[0.5] * 13, std=[0.5] * 13)  # Assuming 13 bands
])

dataset = EuroSATDataset("E:/Thesis/ds", transform=data_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, Loss, Optimizer
num_classes = 10  # EuroSAT has 10 classes
num_bands = 13  # 13 spectral bands
model = CustomCNN(in_channels=num_bands, num_classes=num_classes)  # No .cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        images, labels = images, labels  # No .cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")


# Band Importance Analysis
def compute_band_importance(model, dataloader, num_bands):
    model.eval()
    importance_scores = np.zeros(num_bands)
    with torch.no_grad():
        for images, _ in dataloader:
            images = images  # No .cuda()
            baseline_output = model(images)
            for b in range(num_bands):
                perturbed_images = images.clone()
                perturbed_images[:, b, :, :] = 0  # Zero out one band
                perturbed_output = model(perturbed_images)
                importance_scores[b] += (baseline_output - perturbed_output).abs().mean().item()
    return importance_scores / len(dataloader)


# Compute and Plot Band Importance
importance_scores = compute_band_importance(model, dataloader, num_bands)
plt.figure(figsize=(10, 6))
sns.heatmap(importance_scores.reshape(1, -1), annot=True, cmap='coolwarm', cbar=True,
            xticklabels=[f'Band {i + 1}' for i in range(num_bands)])
plt.title("Class-wise Band Importance Heatmap")
plt.xlabel("Bands")
plt.ylabel("Importance")
plt.show()