import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)

        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.weight is not None:
            loss = loss * self.weight[target]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
from torch.utils.data import WeightedRandomSampler, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# -------------------------
# Preprocessed loaders & dataset
# -------------------------
from preprocess import train_df, val_loader, test_loader, class_weights, img_folder, train_transform
from preprocess import AutoImageDataset

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Parameters
# -------------------------
num_classes = 5
learning_rate = 1e-4
num_epochs = 20
batch_size = 64

# -------------------------
# Weighted Random Sampler for Oversampling Rare Classes
# -------------------------
train_labels = train_df['diagnosis'].values
classes = np.unique(train_labels)

# Count samples per class
class_sample_counts = np.array([sum(train_labels == i) for i in classes])
weights_per_class = 1. / class_sample_counts

# Assign weight to each sample
samples_weight = np.array([weights_per_class[t] for t in train_labels])
samples_weight = torch.from_numpy(samples_weight).float()
weights_per_class[2] *= 2.5 # Severe
weights_per_class[3] *= 2.5  #P ROLIFERATE dr
weights_per_class[4]*=1.8
samples_weight = np.array([weights_per_class[t] for t in train_labels])
samples_weight = torch.from_numpy(samples_weight).float()
# Weighted sampler
sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

# Train dataset & loader
train_dataset = AutoImageDataset(train_df, img_folder, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True)

# -------------------------
# Model
# -------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

optimizer = torch.optim.Adam([
    {'params': model.layer2.parameters(), 'lr': 1e-5},  # subtle layers
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-4}        # classifier head higher LR
])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Gradual unfreeze last 2 blocks + fc
# ---------------------------
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True
# -------------------------
# Loss & Optimizer
# -------------------------
weights = torch.tensor([1.0, 1.0, 2.3, 2.5, 1.8], dtype=torch.float).to(device)
criterion = FocalLoss(weight=None, gamma=2.0) 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------
# Training Loop
# -------------------------
best_acc = 0.0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 30)

    for phase, loader in [("train", train_loader), ("val", val_loader)]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0
        total = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == "train":
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels)
            total += labels.size(0)
        if phase == "train":
            scheduler.step()    
        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total

        print(f"{phase.upper()} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        # Save best model
        if phase == "val" and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), "best_resnet18_model.pth")
            print("** Best model saved **")

print(f"\nTraining complete. Best Validation Accuracy: {best_acc:.4f}")

# -------------------------
# Confusion Matrix on Test Set
# -------------------------
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

# -------------------------
# Plot Confusion Matrix
# -------------------------
class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Diabetic Retinopathy")
plt.show()
