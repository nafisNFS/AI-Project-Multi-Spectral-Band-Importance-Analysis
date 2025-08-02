import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import shap
import rasterio
from torch_optimizer import Lamb

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
                with rasterio.open(image_path) as src:
                    image = src.read()  # Read all bands
                    image = np.transpose(image, (1, 2, 0))  # Change to (height, width, bands)
                    image = image[:, :, :3]  # Use only the first 3 bands (RGB)
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

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert numpy array to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

# Load the dataset
dataset_path = r"E:\Thesis\EuroSATallBands"
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

# Step 2: Build the Custom CNN with ReLU
class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),  # Use ReLU
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),  # Use ReLU
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),  # Use ReLU
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),  # Use ReLU
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),  # Use ReLU
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),  # Use ReLU
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),  # Use ReLU
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),  # Use ReLU
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),  # Use ReLU
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),  # Use ReLU
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(),  # Use ReLU
            nn.Linear(512, 256),
            nn.ReLU(),  # Use ReLU
            nn.Linear(256, 128),
            nn.ReLU(),  # Use ReLU
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Lamb(model.parameters(), lr=1e-3)  # Use LAMB optimizer

# Step 3: Train the Model with Early Stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=1, patience=5):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')
    epochs_without_improvement = 0

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

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return train_losses, val_losses, train_accs, val_accs

# Train the model
train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, criterion, optimizer)

# Step 4: Evaluate the Model on Test Set
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Evaluate the model
evaluate_model(model, test_loader)

# Step 5: SHAP Global Explanations with DeepExplainer
def shap_global_explanations(model, test_loader):
    model.eval()

    # Use fewer samples for background and test images
    background = next(iter(test_loader))[0][:5].to(device)  # Use 5 samples as background
    test_images = next(iter(test_loader))[0][:1].to(device)  # Use 1 sample for explanation

    # Use DeepExplainer
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_images)

    # Define proper band names and class names
    band_names = ["B01 - Aerosols", "B02 - Blue", "B03 - Green", "B04 - Red",
                  "B05 - Red Edge 1", "B06 - Red Edge 2", "B07 - Red Edge 3",
                  "B08 - NIR", "B09 - Water Vapor", "B11 - SWIR 1", "B12 - SWIR 2"]
    class_names = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
                   "Industrial", "Pasture", "PermanentCrop", "Residential",
                   "River", "SeaLake"]

    # Plot SHAP global explanations as a bar plot
    shap.summary_plot(shap_values, test_images.cpu().numpy(), feature_names=band_names, class_names=class_names, plot_type="bar")
# Generate SHAP global explanations
shap_global_explanations(model, test_loader)

# Step 6: Plot Training and Validation Metrics
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()