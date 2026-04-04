# evaluate.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Import your modules
from model import DiabeticRetinopathyModel
# test.py (or evaluate.py)
from preprocess import get_loader,test_df,img_folder, AutoImageDataset  # replace with your function to get test dataset

# ------------------------------
# Configuration
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5
batch_size = 16  # adjust according to your GPU memory

# ------------------------------
# Load model
# ------------------------------
model = DiabeticRetinopathyModel(num_classes=num_classes)
model_path = r"C:\Users\user\Desktop\Diabetic retinopathy model\best_resnet_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
model.to(device)
model.eval()

# ------------------------------
# Load test dataset
# ------------------------------
# test_dataset = get_test_dataset()  # should return a PyTorch Dataset
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_loader = get_loader(test_df, img_folder, batch_size=64, train=False)

# ------------------------------
# Run inference
# ------------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ------------------------------
# Confusion matrix
# ------------------------------
cm = confusion_matrix(all_labels, all_preds)
classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Diabetic Retinopathy")
plt.show()

# ------------------------------
# Metrics       
# ------------------------------
print("Accuracy:", accuracy_score(all_labels, all_preds))
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=classes))