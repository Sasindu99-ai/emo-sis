"""Facial expression classification model architecture.

Implements MobileNetV2 transfer learning backbone (with fine-tuning head)
for patient distress expression classification (neutral, pain, panic, agitation).
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

DEFAULT_FACIAL_CLASSES = [
    "neutral",
    "pain",
    "panic",
    "agitation"
]


class FacialEmotionModel(nn.Module):
    """MobileNetV2 based classifier for patient distress facial expression recognition."""

    def __init__(self, num_classes: int = len(DEFAULT_FACIAL_CLASSES), pretrained: bool = False, dropout_rate: float = 0.3):
        super(FacialEmotionModel, self).__init__()
        
        # Lightweight MobileNetV2 backbone
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        try:
            self.backbone = models.mobilenet_v2(weights=weights)
        except Exception:
            self.backbone = models.mobilenet_v2(weights=None)
            
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (batch_size, 3, 224, 224)
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
