# anaemia_detection/models/anemia_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class EfficientNetAnemia(nn.Module):
    def __init__(self, num_classes=2, model_name='efficientnet_b0'):
        super(EfficientNetAnemia, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)

        # Freeze all, unfreeze last 4 blocks + head
        for param in self.backbone.parameters():
            param.requires_grad = False
        for name, param in self.backbone.named_parameters():
            if any(f'blocks.{i}' in name for i in [3, 4, 5, 6]):
                param.requires_grad = True
            if any(k in name for k in ['conv_head', 'bn2', 'global_pool']):
                param.requires_grad = True

        feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    # Remove get_param_groups entirely

class ResNetAnemia(nn.Module):
    """Using ResNet as backbone"""
    def __init__(self, num_classes=2, model_name='resnet18'):
        super(ResNetAnemia, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        feature_dim = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def create_model(model_type='simple', num_classes=2):
    """Factory function to create different models"""
    if model_type == 'simple':
        return SimpleCNN(num_classes=num_classes)
    elif model_type == 'efficientnet':
        return EfficientNetAnemia(num_classes=num_classes)
    elif model_type == 'resnet':
        return ResNetAnemia(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")