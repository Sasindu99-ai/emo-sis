"""Training pipeline for the Facial Emotion / Distress branch.

Fine-tunes FacialEmotionModel (MobileNetV2) on facial datasets,
evaluates validation accuracy and confusion matrix, and checkpoints the best model.
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from ai.facial.model import FacialEmotionModel, DEFAULT_FACIAL_CLASSES


class FacialDataset(Dataset):
    """Dataset for facial image paths and mapped distress classes."""

    def __init__(self, file_paths: List[str], labels: List[int], transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(file_path).convert("RGB")
            tensor = self.transform(image)
        except Exception:
            tensor = torch.zeros(3, 224, 224, dtype=torch.float32)
        return tensor, label


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    acc = float(np.mean(all_preds == all_targets)) if len(all_targets) > 0 else 0.0

    return acc, all_targets, all_probs


def train(
    data_dir: str,
    output_dir: str = "ai/models",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_path / "facial_model.pt"

    classes = DEFAULT_FACIAL_CLASSES
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Scan directory
    root = Path(data_dir)
    file_paths = []
    labels = []

    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for path in root.rglob(ext):
            folder_name = path.parent.name.lower()
            for cls_name in classes:
                if cls_name in folder_name:
                    file_paths.append(str(path))
                    labels.append(class_to_idx[cls_name])
                    break

    if len(file_paths) == 0:
        print(f"No labeled images found in {data_dir}. Please populate ai/facial/data per README.md.")
        return

    # Train / Val split
    indices = np.random.permutation(len(file_paths))
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = FacialDataset([file_paths[i] for i in train_idx], [labels[i] for i in train_idx])
    val_ds = FacialDataset([file_paths[i] for i in val_idx], [labels[i] for i in val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = FacialEmotionModel(num_classes=len(classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / max(1, len(train_ds))
        val_acc, val_targets, val_probs = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d}/{epochs} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "classes": classes,
                "state_dict": model.state_dict(),
                "val_acc": val_acc
            }, checkpoint_file)
            print(f"  -> Checkpoint saved to {checkpoint_file} (Val Acc: {val_acc:.4f})")

    print("\nFinal Facial Model Evaluation:")
    val_acc, val_targets, val_probs = evaluate(model, val_loader, device)
    val_preds = np.argmax(val_probs, axis=-1)
    print("Confusion Matrix:")
    print(confusion_matrix(val_targets, val_preds))
    print("\nClassification Report:")
    print(classification_report(val_targets, val_preds, target_names=classes, zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FacialEmotionModel")
    parser.add_argument("--data_dir", type=str, default="ai/facial/data", help="Path to facial data directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    train(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
