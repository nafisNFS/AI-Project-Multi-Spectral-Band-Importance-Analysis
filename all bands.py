import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import rasterio  # For reading TIFF images
from torch_optimizer import Lamb  # LAMB optimizer
import json  # For loading label_map.json

# Step 1: Load and Preprocess the Dataset
class EuroSATDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.class_names = sorted(os.listdir(dataset_path))
        self.transform = transform
        self.images = []
        self.labels = []

        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(dataset_path, class_name)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                # Use rasterio to read the TIFF image
                with rasterio.open(image_path) as src:
                    image = src.read()  # Read all bands
                    image = np.transpose(image, (1, 2, 0))  # Change from (bands, height, width) to (height, width, bands)
                    image = (image / 255.0).astype(np.float32)  # Normalize to [0, 1]
                    self.images.append(image)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Function to calculate mean and std for the dataset
def calculate_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    mean = 0.0
    std = 0.0
    num_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        num_samples += batch_samples

    mean /= num_samples
    std /= num_samples
    return mean, std

# Load the dataset without normalization
dataset_path = r"E:\Thesis\ds"
dataset = EuroSATDataset(dataset_path, transform=transforms.ToTensor())

# Calculate mean and std for the dataset
mean, std = calculate_mean_std(dataset)
print("Mean:", mean)
print("Std:", std)

# Define transformations with calculated mean and std
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert numpy array to tensor
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())  # Normalize using calculated mean and std
])

# Reload the dataset with normalization
dataset = EuroSATDataset(dataset_path, transform=transform)

# Split the dataset into 70% train, 10% validation, and 20% test
train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)

train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset = torch.utils.data.Subset(dataset, val_idx)
test_dataset = torch.utils.data.Subset(dataset, test_idx)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 2: Build the Custom CNN with GELU
class CustomCNN(nn.Module):
    def __init__(self, num_classes=10, num_bands=13):  # Update num_bands to match the number of bands
        super(CustomCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_bands, 32, kernel_size=3, padding=1),  # Update input channels to num_bands
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )

        # Store the activations and gradients
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.conv_layers(x)
        if x.requires_grad:  # Only register the hook if x requires gradients
            h = x.register_hook(self.activations_hook)  # Register hook to store gradients
        self.activations = x  # Store activations
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.activations

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN(num_classes=10, num_bands=13).to(device)  # Update num_bands to match the number of bands
criterion = nn.CrossEntropyLoss()
optimizer = Lamb(model.parameters(), lr=1e-3)  # Use LAMB optimizer

# Step 3: Train the Model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    return train_losses, val_losses, train_accs, val_accs

# Train the model
train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, criterion, optimizer)

# Step 4: Evaluate the Model on Test Set
def evaluate_model(model, test_loader, label_map):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Class-wise precision, recall, and F1-score
    class_report = classification_report(y_true, y_pred, target_names=list(label_map.keys()), output_dict=True)
    print("Class-wise Metrics:")
    for class_name, metrics in class_report.items():
        if class_name in label_map:  # Skip 'accuracy', 'macro avg', 'weighted avg'
            print(f"Class: {class_name}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1-score']:.4f}")

    # Final accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("\nFinal Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Load the label map
label_map_path = r"E:\Thesis\bands info\label_map.json"
with open(label_map_path, 'r') as f:
    label_map = json.load(f)

# Evaluate the model
evaluate_model(model, test_loader, label_map)

# Step 5: Grad-CAM for Band Importance
def grad_cam(model, input_image, target_class):
    model.eval()
    input_image = input_image.unsqueeze(0).to(device)  # Add batch dimension

    # Forward pass
    output = model(input_image)
    model.zero_grad()

    # Backward pass for the target class
    output[:, target_class].backward()

    # Get activations and gradients
    gradients = model.get_activations_gradient()  # Shape: (batch_size, channels, height, width)
    activations = model.get_activations()  # Shape: (batch_size, channels, height, width)

    # Pool the gradients across the spatial dimensions (height, width)
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])  # Shape: (channels,)

    # Weight the activations by the pooled gradients
    for i in range(activations.shape[1]):  # Iterate over channels
        activations[:, i, :, :] *= pooled_gradients[i]

    # Average the weighted activations across the channels
    heatmap = torch.mean(activations, dim=1).squeeze().cpu()  # Shape: (height, width)

    # Apply ReLU to the heatmap
    heatmap = torch.relu(heatmap)

    # Normalize the heatmap
    heatmap /= torch.max(heatmap)

    return heatmap

# Reverse the label map to get class names from indices
class_names = {v: k for k, v in label_map.items()}

# Define proper band names
band_names = [
    "B01 - Aerosols", "B02 - Blue", "B03 - Green", "B04 - Red",
    "B05 - Red edge 1", "B06 - Red edge 2", "B07 - Red edge 3",
    "B08 - NIR", "B08A - Red edge 4", "B09 - Water vapor",
    "B10 - Cirrus", "B11 - SWIR 1", "B12 - SWIR 2"
]

def visualize_band_importance(model, test_loader, num_classes=10):
    model.eval()
    images, labels = next(iter(test_loader))
    images = images.to(device)

    # Create a figure to plot the heatmaps
    plt.figure(figsize=(15, 8))

    for target_class in range(num_classes):
        # Compute Grad-CAM for the target class
        heatmap = grad_cam(model, images[0], target_class)

        # Detach the heatmap and convert to NumPy
        heatmap = heatmap.detach().cpu().numpy()

        # Aggregate the heatmap to match the number of bands (13)
        # Here, we assume the heatmap corresponds to the 13 bands
        band_importance = np.mean(heatmap, axis=(0, 1))  # Aggregate spatial dimensions

        # Ensure the band_importance has the correct shape (13,)
        if band_importance.shape != (13,):
            raise ValueError(f"Band importance shape mismatch: expected (13,), got {band_importance.shape}")

        # Plot the heatmap as a bar plot
        plt.subplot(2, 5, target_class + 1)
        plt.bar(range(len(band_names)), band_importance, color='skyblue')
        plt.title(f'Class: {class_names[target_class]}', fontsize=10)
        plt.xlabel('Bands', fontsize=8)
        plt.ylabel('Importance', fontsize=8)
        plt.xticks(ticks=range(len(band_names)), labels=band_names, rotation=90, fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

# Visualize band importance for each class
visualize_band_importance(model, test_loader)