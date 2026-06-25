import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# =====================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# =====================================================================
# CONFIGURATION: Change this to the exact path of your EuroSAT RGB folder
DATASET_ROOT = "C:/Users/Windows 11 Pro/Videos/sat/EuroSAT" 

# Hardware targeting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Hyperparameters (Optimized for local RTX 5070 execution)
BATCH_SIZE = 128  # Scaled up to maximize your 5070 VRAM
INITIAL_LR = 0.001
FINE_TUNE_LR = 0.0001
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 5
NUM_CLASSES = 10

# ImageNet normalization statistics expected by ResNet50
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

class_names = ['AnnualCrop', 'Forest', 'HerbVeg', 'Highway', 'Industrial', 
               'Pasture', 'PermCrop', 'Residential', 'River', 'SeaLake']


# =====================================================================
# 2. DATA PREPROCESSING & DATA LOADERS
# =====================================================================
print("\n--- Setting Up Transforms and Data Loaders ---")

# Training transforms include Data Augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)), # ResNet50 requires 224x224
    transforms.RandomHorizontalFlip(), # Satellite data has no true orientation
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# Validation/Test data uses standard preprocessing without augmentation
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# Wrapper class to safely apply different transforms to the dataset splits
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        # 1. Map the split index back to the true original dataset index
        actual_idx = self.subset.indices[index]
        
        # 2. Grab the correct image path and label
        path, y = self.subset.dataset.samples[actual_idx]
        
        # 3. Load the image and apply the split-specific transform
        x = Image.open(path).convert('RGB')
        if self.transform:
            x = self.transform(x)
            
        return x, y
        
    def __len__(self):
        return len(self.subset)


# =====================================================================
# 4. TRAINING & EVALUATION HELPER FUNCTIONS
# =====================================================================
def run_epoch(model, dataloader, criterion, optimizer=None, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()
        
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.set_grad_enabled(is_train):
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            if is_train:
                optimizer.zero_grad()
                
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            if is_train:
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# =====================================================================
# 8. PRODUCTION DEFORESTATION CHANGE DETECTION ENGINE
# =====================================================================
def detect_deforestation(img_path_year1, img_path_year2, inference_transform, model, patch_size=64):
    """
    Chunks two historical target full-scale geographic images, evaluates changes, 
    and returns localized map coordinates where tree loss is observed.
    """
    print(f"\n--- Initiating Change Detection Engine: {img_path_year1} vs {img_path_year2} ---")
    img1 = Image.open(img_path_year1).convert('RGB')
    img2 = Image.open(img_path_year2).convert('RGB')
    
    assert img1.size == img2.size, "Error: Spatial scales between both operational images must match exactly."
    width, height = img1.size
    
    model.eval()
    forest_idx = class_names.index('Forest')
    deforestation_events = []
    
    for i in range(0, width - patch_size, patch_size):
        for j in range(0, height - patch_size, patch_size):
            
            patch1 = img1.crop((i, j, i + patch_size, j + patch_size))
            patch2 = img2.crop((i, j, i + patch_size, j + patch_size))
            
            t1 = inference_transform(patch1).unsqueeze(0).to(DEVICE)
            t2 = inference_transform(patch2).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                out1 = model(t1)
                out2 = model(t2)
                _, pred1 = torch.max(out1, 1)
                _, pred2 = torch.max(out2, 1)
            
            p1_class = pred1.item()
            p2_class = pred2.item()
            
            if p1_class == forest_idx and p2_class != forest_idx:
                deforestation_events.append({
                    "grid_coord": (i, j),
                    "converted_to": class_names[p2_class]
                })
                
    print(f"Analysis Complete. Detected {len(deforestation_events)} regional deforestation events.")
    return deforestation_events


# =====================================================================
# RUN TIME EXECUTION GUARD (Protects imports from resetting training)
# =====================================================================
if __name__ == "__main__":
    
    # Load full dataset using folder names as labels
    full_dataset = datasets.ImageFolder(root=DATASET_ROOT, transform=train_transform)

    # Splitting dataset: 80% Train, 10% Validation, 10% Test
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    raw_train, raw_val, raw_test = random_split(full_dataset, [train_size, val_size, test_size])

    train_set = TransformedDataset(raw_train, transform=train_transform)
    val_set = TransformedDataset(raw_val, transform=test_transform)
    test_set = TransformedDataset(raw_test, transform=test_transform)

    # Initialize Python Data Loaders for batch streaming
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    print("\n--- Constructing Model Architecture ---")
    # Load Pretrained ResNet50
    model = models.resnet50(pretrained=True)

    # Freeze all lower layer weights to preserve general feature extractions
    for param in model.parameters():
        param.requires_grad = False

    # Swap out final fully connected layer for EuroSAT classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)

    # Offload model parameters onto local GPU
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    # Pass only the newly configured top classification layer to optimizer
    optimizer = optim.Adam(model.fc.parameters(), lr=INITIAL_LR)

    # =====================================================================
    # 5. EXECUTE INITIAL COARSE TRAINING
    # =====================================================================
    print("\n--- Phase 1: Training Final Classification Layer ---")
    for epoch in range(INITIAL_EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, is_train=False)
        
        print(f"Epoch {epoch+1}/{INITIAL_EPOCHS} -> "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # =====================================================================
    # 6. EXECUTE FINE-TUNING DEEPER LAYERS
    # =====================================================================
    print("\n--- Phase 2: Unfreezing Model & Fine-Tuning All Layers ---")
    # Unfreeze all foundational ResNet convolutional parameters
    for param in model.parameters():
        param.requires_grad = True

    # Lower learning rate dramatically to preserve feature integrity
    optimizer = optim.Adam(model.parameters(), lr=FINE_TUNE_LR)

    for epoch in range(FINE_TUNE_EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, is_train=False)
        
        print(f"Fine-Tune Epoch {epoch+1}/{FINE_TUNE_EPOCHS} -> "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save core weights locally
    torch.save(model.state_dict(), "local_resnet50_eurosat.pth")
    print("Saved trained weights locally to 'local_resnet50_eurosat.pth'")

    # =====================================================================
    # 7. MODEL EVALUATION & ANALYSIS METRICS
    # =====================================================================
    print("\n--- Running Evaluation on Hold-Out Test Split ---")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Construct and Save Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('EuroSAT Local Model Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Generated evaluation breakdown matrix saved as 'confusion_matrix.png'")
    plt.close()