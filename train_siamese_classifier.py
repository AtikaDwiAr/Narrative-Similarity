"""
Training script for Siamese Classifier with early stopping and validation monitoring.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from pathlib import Path
from tqdm import tqdm
import numpy as np

from dataset_classifier import create_dataloaders
from model_siamese_classifier import SiameseClassifier
from utils import (
    set_seed, get_device, compute_metrics, 
    print_metrics_table, print_training_summary
)


class SiameseTrainer:
    """
    Trainer class for Siamese Classifier.
    """
    
    def __init__(
        self,
        model: SiameseClassifier,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str,
        output_dir: str = "output",
        patience: int = 5
    ):
        """
        Initialize trainer.
        
        Args:
            model: SiameseClassifier model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to use ("cuda" or "cpu")
            output_dir: Directory to save models
            patience: Early stopping patience
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.output_dir = Path(output_dir)
        self.patience = patience
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_acc = 0
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_epoch = 0
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            anchor, text_a, text_b, labels = batch
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model.forward(anchor, text_a, text_b)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self) -> tuple:
        """
        Validate on validation set.
        
        Returns:
            Tuple of (val_loss, val_accuracy)
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for batch in progress_bar:
                anchor, text_a, text_b, labels = batch
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model.forward(anchor, text_a, text_b)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        return avg_loss, accuracy
    
    def train(self, num_epochs: int) -> None:
        """
        Train model for specified number of epochs with early stopping.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Patience: {self.patience}")
        print(f"Output directory: {self.output_dir}\n")
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_acc:.4f}")
            
            # Check if best
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(epoch, val_acc, is_best=True)
                print(f"✓ New best validation accuracy: {val_acc:.4f}")
            else:
                self.patience_counter += 1
                print(f"No improvement. Patience: {self.patience_counter}/{self.patience}")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        # Print training summary
        print_training_summary(
            len(self.train_losses),
            self.train_losses,
            self.val_losses,
            self.val_accuracies,
            self.best_epoch
        )
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_acc: Validation accuracy
            is_best: Whether this is the best model
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies
        }
        
        if is_best:
            path = self.output_dir / "siamese_classifier_best.pth"
            torch.save(checkpoint, path)
            print(f"Saved best model to {path}")


def main():
    """
    Main training function.
    """
    # Set random seeds
    set_seed(42)
    
    # Get device
    device = get_device()
    
    # Hyperparameters
    CONFIG = {
        "sbert_model_path": "all-MiniLM-L6-v2",
        "data_path": "data/train_track_a.jsonl",
        "train_ratio": 0.8,
        "batch_size": 16,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "num_epochs": 20,
        "early_stopping_patience": 5,
        "hidden_dim": 256,
        "dropout_rate": 0.2,
        "freeze_sbert": True,
        "output_dir": "output"
    }
    
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"{'='*60}")
    for key, value in CONFIG.items():
        print(f"{key:<25}: {value}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading datasets...")
    train_loader, val_loader = create_dataloaders(
        CONFIG["data_path"],
        train_ratio=CONFIG["train_ratio"],
        batch_size=CONFIG["batch_size"]
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}\n")
    
    # Initialize model
    print("Initializing model...")
    model = SiameseClassifier(
        sbert_model_path=CONFIG["sbert_model_path"],
        hidden_dim=CONFIG["hidden_dim"],
        dropout_rate=CONFIG["dropout_rate"],
        freeze_sbert=CONFIG["freeze_sbert"]
    )
    
    # Print model info
    param_counts = model.count_parameters()
    print(f"Model parameters:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Trainable: {param_counts['trainable']:,}")
    print(f"  Frozen: {param_counts['frozen']:,}\n")
    
    # Setup optimizer and loss
    optimizer = AdamW(
        model.get_trainable_params(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss()
    
    # Initialize trainer
    trainer = SiameseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        output_dir=CONFIG["output_dir"],
        patience=CONFIG["early_stopping_patience"]
    )
    
    # Train
    trainer.train(num_epochs=CONFIG["num_epochs"])
    
    print("\n✓ Training completed!")
    print(f"Best model saved to {CONFIG['output_dir']}/siamese_classifier_best.pth")


if __name__ == "__main__":
    main()
