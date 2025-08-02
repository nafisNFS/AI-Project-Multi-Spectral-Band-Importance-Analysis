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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset


# Step 1: Load and Preprocess the Dataset
class EuroSATDataset(Dataset):
    def __init__(self, dataset_path, band_indices, transform=None):
        self.dataset_path = dataset_path
        self.class_names = sorted(os.listdir(dataset_path))
        self.transform = transform
        self.band_indices = band_indices
        self.images = []
        self.labels = []

        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(dataset_path, class_name)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                with rasterio.open(image_path) as src:
                    image = src.read()
                    image = np.transpose(image, (1, 2, 0))
                    image = (image / 255.0).astype(np.float32)

                    if band_indices is None:
                        nir = image[:, :, 7]
                        red = image[:, :, 3]
                        swir = image[:, :, 11]

                        ndvi = (nir - red) / (nir + red + 1e-10)
                        ndbi = (swir - nir) / (swir + nir + 1e-10)
                        ndwi = (nir - swir) / (nir + swir + 1e-10)

                        image = np.stack([ndbi, ndvi, ndwi], axis=-1)
                    else:
                        image = image[:, :, band_indices]

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
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define band combinations
band_combinations = {
    "SWIR-NIR-RED": [11, 7, 3],
    "NIR-RED-GREEN": [7, 3, 2],
    "NDBI-NDVI-NDWI": None
}


# Step 2: Build the Custom CNN with GELU
class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
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

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# Step 3: Train and Evaluate for Each Band Combination
def train_and_evaluate(band_combination_name, band_indices):
    print(f"Training and evaluating for band combination: {band_combination_name}")

    dataset_path = r"E:\Thesis\EuroSATallBands"
    dataset = EuroSATDataset(dataset_path, band_indices, transform=transform)

    train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Lamb(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    best_val_loss = float('inf')
    patience = 10
    early_stopping_counter = 0

    for epoch in range(15):
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

        print(f"Epoch [{epoch + 1}/15], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step(val_loss)

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

    # Class-wise precision, recall, and F1-score
    classwise_precision = precision_score(y_true, y_pred, average=None)
    classwise_recall = recall_score(y_true, y_pred, average=None)
    classwise_f1 = f1_score(y_true, y_pred, average=None)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nClass-wise Metrics:")
    for i, (prec, rec, f1_score_class) in enumerate(zip(classwise_precision, classwise_recall, classwise_f1)):
        print(f"Class {i}: Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1_score_class:.4f}")

    # SHAP Analysis with GradientExplainer
    background = torch.cat([inputs for inputs, _ in train_loader], dim=0)[:100].to(device)
    test_images = torch.cat([inputs for inputs, _ in test_loader], dim=0)[:10].to(device)

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(test_images)

    return accuracy, precision, recall, f1, classwise_precision, classwise_recall, classwise_f1, shap_values, test_images


# Step 4: Train and evaluate for each band combination
results = {}
shap_outputs = {}  # Store SHAP outputs for later visualization

for band_combination_name, band_indices in band_combinations.items():
    accuracy, precision, recall, f1, classwise_precision, classwise_recall, classwise_f1, shap_values, test_images = train_and_evaluate(
        band_combination_name, band_indices)
    results[band_combination_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Classwise Precision": classwise_precision,
        "Classwise Recall": classwise_recall,
        "Classwise F1 Score": classwise_f1
    }
    shap_outputs[band_combination_name] = (shap_values, test_images)  # Store SHAP outputs

# Print results for each band combination
print("\nResults for each band combination:")
for band_combination_name, metrics in results.items():
    print(f"\nBand Combination: {band_combination_name}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1 Score: {metrics['F1 Score']:.4f}")
    print("\nClass-wise Metrics:")
    for i, (prec, rec, f1_score_class) in enumerate(
            zip(metrics['Classwise Precision'], metrics['Classwise Recall'], metrics['Classwise F1 Score'])):
        print(f"Class {i}: Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1_score_class:.4f}")

# Display SHAP outputs at the very end
print("\nDisplaying SHAP outputs for each band combination:")
for band_combination_name, (shap_values, test_images) in shap_outputs.items():
    print(f"\nSHAP Output for Band Combination: {band_combination_name}")

    # Plot SHAP values
    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)

    # Ensure the SHAP values and test images have the same shape
    shap_numpy = [np.mean(s, axis=-1) for s in shap_numpy]  # Take the mean across the channels
    test_numpy = np.mean(test_numpy, axis=-1)  # Take the mean across the channels

    shap.image_plot(shap_numpy, -test_numpy)