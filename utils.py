"""
Utility functions for Siamese classifier training and evaluation.
"""

import json
from pathlib import Path
from typing import List, Dict, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_jsonl(filepath: str) -> List[dict]:
    """
    Load data from JSONL file.
    
    Args:
        filepath: Path to JSONL file
        
    Returns:
        List of dictionaries loaded from JSONL
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    
    return data


def save_jsonl(data: List[dict], filepath: str) -> None:
    """
    Save data to JSONL file.
    
    Args:
        data: List of dictionaries to save
        filepath: Path to save JSONL file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Saved {len(data)} items to {filepath}")


def convert_labels(labels: np.ndarray) -> np.ndarray:
    """
    Convert label format for evaluation.
    Label 0 (text_a closer) → True
    Label 1 (text_b closer) → False
    
    Args:
        labels: Array of labels (0 or 1)
        
    Returns:
        Array of boolean labels
    """
    return labels == 0


def compute_metrics(
    predictions: Union[List[int], np.ndarray],
    labels: Union[List[int], np.ndarray]
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Predicted labels (0 or 1)
        labels: True labels (0 or 1)
        
    Returns:
        Dictionary with metrics:
        - accuracy: Classification accuracy
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0)
    }
    
    return metrics


def print_metrics_table(
    metrics: Dict[str, float],
    model_name: str = "Model",
    title: str = "Classification Metrics"
) -> None:
    """
    Pretty print metrics in table format.
    
    Args:
        metrics: Dictionary of metric names and values
        model_name: Name of the model
        title: Title for the table
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Value':>15}")
    print(f"{'-'*60}")
    
    for metric_name, value in metrics.items():
        print(f"{metric_name:<20} {value:>15.4f}")
    
    print(f"{'='*60}\n")


def compare_predictions(
    pred1: List[dict],
    pred2: List[dict],
    gold: List[dict],
    pred1_name: str = "Model 1",
    pred2_name: str = "Model 2"
) -> None:
    """
    Compare two sets of predictions against ground truth.
    
    Args:
        pred1: List of prediction dictionaries from first model
        pred2: List of prediction dictionaries from second model
        gold: List of ground truth dictionaries
        pred1_name: Name of first model
        pred2_name: Name of second model
    """
    if not (len(pred1) == len(pred2) == len(gold)):
        raise ValueError("All prediction sets must have same length")
    
    # Extract predictions and labels
    labels = np.array([g["text_a_is_closer"] for g in gold])
    preds1 = np.array([p["text_a_is_closer"] for p in pred1])
    preds2 = np.array([p["text_a_is_closer"] for p in pred2])
    
    # Convert boolean to integer labels for metrics
    preds1_int = (~preds1).astype(int)  # True→0, False→1
    preds2_int = (~preds2).astype(int)
    labels_int = (~labels).astype(int)
    
    # Compute metrics
    metrics1 = compute_metrics(preds1_int, labels_int)
    metrics2 = compute_metrics(preds2_int, labels_int)
    
    # Print comparison table
    print(f"\n{'='*80}")
    print(f"{'PREDICTION COMPARISON':^80}")
    print(f"{'='*80}")
    print(f"{'Metric':<20} {pred1_name:>25} {pred2_name:>25}")
    print(f"{'-'*80}")
    
    for metric_name in ["accuracy", "precision", "recall", "f1"]:
        v1 = metrics1[metric_name]
        v2 = metrics2[metric_name]
        diff = v2 - v1
        diff_sign = "+" if diff >= 0 else ""
        
        print(f"{metric_name:<20} {v1:>25.4f} {v2:>25.4f}")
        print(f"{'  ' + metric_name + ' diff':<20} {diff_sign}{diff:>24.4f}")
    
    print(f"{'='*80}\n")
    
    # Count agreements/disagreements
    agreement = (preds1 == preds2).sum()
    disagreement = (preds1 != preds2).sum()
    
    print(f"Predictions agreement: {agreement}/{len(preds1)} ({100*agreement/len(preds1):.2f}%)")
    print(f"Predictions disagreement: {disagreement}/{len(preds1)} ({100*disagreement/len(preds1):.2f}%)")
    
    # Count correctness
    correct1 = (preds1 == labels).sum()
    correct2 = (preds2 == labels).sum()
    
    print(f"\n{pred1_name} correct: {correct1}/{len(labels)}")
    print(f"{pred2_name} correct: {correct2}/{len(labels)}")


def print_training_summary(
    epochs: int,
    train_losses: List[float],
    val_losses: List[float],
    val_accuracies: List[float],
    best_epoch: int
) -> None:
    """
    Print training summary.
    
    Args:
        epochs: Total number of epochs
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        val_accuracies: List of validation accuracies per epoch
        best_epoch: Epoch with best validation accuracy
    """
    print(f"\n{'='*80}")
    print(f"{'TRAINING SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"Total epochs: {epochs}")
    print(f"Best epoch: {best_epoch + 1}")
    print(f"Best validation accuracy: {val_accuracies[best_epoch]:.4f}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"\n{'Epoch':<10} {'Train Loss':<15} {'Val Loss':<15} {'Val Acc':<15}")
    print(f"{'-'*80}")
    
    for epoch in range(epochs):
        marker = " ← BEST" if epoch == best_epoch else ""
        print(f"{epoch+1:<10} {train_losses[epoch]:<15.4f} {val_losses[epoch]:<15.4f} {val_accuracies[epoch]:<15.4f}{marker}")
    
    print(f"{'='*80}\n")


def get_device() -> str:
    """
    Get available device (CUDA or CPU).
    
    Returns:
        Device string: "cuda" or "cpu"
    """
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {device}")
    return device


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    print(f"Random seed set to {seed}")


if __name__ == "__main__":
    # Quick test
    print("Testing utils functions...")
    
    # Test load_jsonl
    try:
        data = load_jsonl("data/train_track_a.jsonl")
        print(f"✓ Loaded {len(data)} samples from JSONL")
    except Exception as e:
        print(f"✗ Error loading JSONL: {e}")
    
    # Test compute_metrics
    true_labels = [0, 0, 1, 1, 0]
    pred_labels = [0, 1, 1, 0, 0]
    metrics = compute_metrics(pred_labels, true_labels)
    print(f"\n✓ Computed metrics:")
    print_metrics_table(metrics, "Test Model")
    
    # Test device detection
    device = get_device()
    print(f"✓ Device: {device}")
    
    # Test seed setting
    set_seed(42)
    print("✓ Random seed set")
