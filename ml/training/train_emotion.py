"""
EmotionNet Training Script.

Trains EfficientNet-B0 on FER2013 for emotion classification with
multi-task learning (emotion, frustration, engagement).

Usage:
    python train_emotion.py --data_dir ./data/processed/fer2013 --epochs 50

Features:
- Mixed precision training (AMP)
- Cosine annealing with warmup
- Data augmentation (rotation, flip, color jitter)
- Class-weighted loss for imbalanced data
- TensorBoard logging
- Early stopping
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.emotion_net import EmotionNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EmotionTrainer:
    """Trainer class for EmotionNet."""

    def __init__(
        self,
        model: EmotionNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: Path,
        class_weights: torch.Tensor | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir

        # Loss functions with label smoothing for better generalization
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.emotion_criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.1  # Reduces overconfidence
        )
        self.regression_criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler with warmup
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double period after each restart
            eta_min=1e-6,
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        # Tracking
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def train_epoch(self, use_mixup: bool = True) -> dict[str, float]:
        """Train for one epoch with optional Mixup augmentation."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Mixup augmentation
            use_mixup_batch = use_mixup and np.random.random() > 0.5
            if use_mixup_batch:
                lam = np.random.beta(0.4, 0.4)
                rand_idx = torch.randperm(images.size(0)).to(self.device)
                images = lam * images + (1 - lam) * images[rand_idx]
                labels_a, labels_b = labels, labels[rand_idx]

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.scaler:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    if use_mixup_batch:
                        loss = lam * self._compute_loss(outputs, labels_a) + \
                               (1 - lam) * self._compute_loss(outputs, labels_b)
                    else:
                        loss = self._compute_loss(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                if use_mixup_batch:
                    loss = lam * self._compute_loss(outputs, labels_a) + \
                           (1 - lam) * self._compute_loss(outputs, labels_b)
                else:
                    loss = self._compute_loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs["emotions"].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.2f}%",
            )

        return {
            "train_loss": total_loss / len(self.train_loader),
            "train_acc": correct / total,
        }

    def validate(self) -> dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self._compute_loss(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs["emotions"].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return {
            "val_loss": total_loss / len(self.val_loader),
            "val_acc": correct / total,
        }

    def _compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-task loss."""
        # Primary: emotion classification
        emotion_loss = self.emotion_criterion(outputs["emotions"], labels)

        # Auxiliary: frustration/engagement (using emotion labels as proxy)
        # Frustration proxy: angry, fear, sad, disgust -> high frustration
        frustration_target = self._labels_to_frustration(labels)
        engagement_target = self._labels_to_engagement(labels)

        frustration_loss = self.regression_criterion(
            outputs["frustration"].squeeze(),
            frustration_target,
        )
        engagement_loss = self.regression_criterion(
            outputs["engagement"].squeeze(),
            engagement_target,
        )

        # Weighted sum
        total_loss = emotion_loss + 0.3 * frustration_loss + 0.3 * engagement_loss
        return total_loss

    def _labels_to_frustration(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert emotion labels to frustration proxy scores."""
        # angry=0, disgust=1, fear=2, happy=3, sad=4, surprise=5, neutral=6
        frustration_map = torch.tensor(
            [0.8, 0.7, 0.6, 0.1, 0.5, 0.3, 0.2],
            device=labels.device,
        )
        return frustration_map[labels]

    def _labels_to_engagement(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert emotion labels to engagement proxy scores."""
        # Higher for attentive emotions, lower for disengaged
        engagement_map = torch.tensor(
            [0.7, 0.5, 0.6, 0.8, 0.4, 0.9, 0.5],
            device=labels.device,
        )
        return engagement_map[labels]

    def train(
        self,
        epochs: int,
        patience: int = 10,
    ) -> dict[str, list[float]]:
        """
        Full training loop.

        Args:
            epochs: Number of epochs to train
            patience: Early stopping patience

        Returns:
            Dictionary of training history
        """
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_metrics = self.train_epoch()
            history["train_loss"].append(train_metrics["train_loss"])
            history["train_acc"].append(train_metrics["train_acc"])

            # Validate
            val_metrics = self.validate()
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_acc"].append(val_metrics["val_acc"])

            # Update scheduler
            self.scheduler.step()

            # Log metrics
            logger.info(
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_acc']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_acc']:.4f}"
            )

            # Save best model
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.epochs_without_improvement = 0
                self._save_checkpoint("best_model.pt")
                logger.info("Saved best model")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

        return history

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            checkpoint_path,
        )


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Get training and validation transforms."""
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.2),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EmotionNet")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to processed FER2013")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze EfficientNet")
    args = parser.parse_args()

    # Setup
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device selection: prefer MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Data loaders
    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for MPS stability
        pin_memory=False,  # MPS doesn't support pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Load class weights from JSON
    weights_path = data_dir / "class_weights.json"
    class_weights = None
    if weights_path.exists():
        with open(weights_path) as f:
            weights_dict = json.load(f)
        # Map weights to class indices based on ImageFolder class order
        class_names = train_dataset.classes
        weight_list = [weights_dict.get(name, 1.0) for name in class_names]
        class_weights = torch.tensor(weight_list, dtype=torch.float32)
        logger.info(f"Using class weights for {class_names}: {weight_list}")

    # Model
    model = EmotionNet(pretrained=True, freeze_backbone=args.freeze_backbone)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Trainer
    trainer = EmotionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        class_weights=class_weights,
        learning_rate=args.lr,
    )

    # Train
    history = trainer.train(epochs=args.epochs, patience=args.patience)

    logger.info(f"\nTraining complete. Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
