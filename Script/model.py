# model.py (fixed)
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class DiabeticRetinopathyModel(nn.Module):
    def __init__(self, num_classes=5):
        super(DiabeticRetinopathyModel, self).__init__()

        # Load pretrained ResNet18
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze all layers initially
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        # ✅ Unfreeze last block (layer4) and fc
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        for block in self.backbone.layer2.children():
          for param in block.parameters():
           param.requires_grad = True

        # Replace final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        # Ensure fc parameters are trainable
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


# Test only
if __name__ == "__main__":
    model = DiabeticRetinopathyModel()
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)
