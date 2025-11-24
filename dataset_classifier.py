"""
PyTorch Dataset class for loading narrative similarity data from JSONL format.
Supports train/validation split for classification task.
"""

import json
from typing import List, Tuple, Union
from torch.utils.data import Dataset, random_split, DataLoader
from pathlib import Path


class NarrativeSimilarityDataset(Dataset):
    """
    PyTorch Dataset for narrative similarity classification.
    
    Loads data from JSONL format and provides (anchor, text_a, text_b, label) tuples.
    Label encoding: 0 = text_a is closer, 1 = text_b is closer
    """
    
    def __init__(self, jsonl_path: str):
        """
        Initialize dataset by loading JSONL file.
        
        Args:
            jsonl_path: Path to JSONL file containing narrative similarity data
            
        Raises:
            FileNotFoundError: If JSONL file doesn't exist
            ValueError: If required fields are missing from data
        """
        self.jsonl_path = Path(jsonl_path)
        
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")
        
        self.data = self._load_jsonl()
        print(f"Loaded {len(self.data)} samples from {jsonl_path}")
    
    def _load_jsonl(self) -> List[dict]:
        """
        Load data from JSONL file.
        
        Returns:
            List of data dictionaries
            
        Raises:
            ValueError: If required fields are missing
        """
        data = []
        required_fields = {"anchor_text", "text_a", "text_b", "text_a_is_closer"}
        
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                try:
                    sample = json.loads(line)
                    
                    # Validate required fields
                    if not required_fields.issubset(sample.keys()):
                        missing = required_fields - set(sample.keys())
                        print(f"Warning: Line {line_idx} missing fields: {missing}. Skipping.")
                        continue
                    
                    # Skip if any text is empty/None
                    if not all([
                        sample["anchor_text"],
                        sample["text_a"],
                        sample["text_b"]
                    ]):
                        print(f"Warning: Line {line_idx} contains empty texts. Skipping.")
                        continue
                    
                    data.append(sample)
                except json.JSONDecodeError as e:
                    print(f"Warning: Line {line_idx} is not valid JSON: {e}. Skipping.")
                    continue
        
        if not data:
            raise ValueError("No valid data loaded from JSONL file")
        
        return data
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[str, str, str, int]:
        """
        Get a single sample.
        
        Args:
            idx: Index of sample
            
        Returns:
            Tuple of (anchor_text, text_a, text_b, label)
            where label: 0 = text_a closer, 1 = text_b closer
        """
        sample = self.data[idx]
        
        anchor = sample["anchor_text"]
        text_a = sample["text_a"]
        text_b = sample["text_b"]
        
        # Label encoding: True (text_a closer) → 0, False (text_b closer) → 1
        label = 0 if sample["text_a_is_closer"] else 1
        
        return anchor, text_a, text_b, label


def create_dataloaders(
    jsonl_path: str,
    train_ratio: float = 0.8,
    batch_size: int = 16,
    shuffle_train: bool = True,
    num_workers: int = 0,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file
        train_ratio: Ratio for train/val split (default: 0.8)
        batch_size: Batch size for dataloaders
        shuffle_train: Whether to shuffle training data
        num_workers: Number of workers for data loading
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Set random seed for reproducibility
    import torch
    torch.manual_seed(random_seed)
    
    # Load full dataset
    dataset = NarrativeSimilarityDataset(jsonl_path)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    print(f"Train/Val split: {train_size}/{val_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    return train_loader, val_loader


def get_test_dataloader(
    jsonl_path: str,
    batch_size: int = 16,
    num_workers: int = 0
) -> DataLoader:
    """
    Create dataloader for test/prediction set.
    
    Args:
        jsonl_path: Path to test JSONL file
        batch_size: Batch size for dataloader
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader for test data
    """
    dataset = NarrativeSimilarityDataset(jsonl_path)
    
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    return test_loader


if __name__ == "__main__":
    # Quick test
    print("Testing dataset loading...")
    
    # Load training dataset
    train_loader, val_loader = create_dataloaders(
        "data/train_track_a.jsonl",
        train_ratio=0.8,
        batch_size=16
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Get one batch as sample
    batch = next(iter(train_loader))
    anchor, text_a, text_b, labels = batch
    
    print(f"\nSample batch:")
    print(f"Anchor: {anchor[0][:100]}...")
    print(f"Text A: {text_a[0][:100]}...")
    print(f"Text B: {text_b[0][:100]}...")
    print(f"Labels: {labels}")
