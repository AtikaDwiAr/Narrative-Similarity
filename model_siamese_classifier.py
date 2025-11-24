"""
Siamese Network with Classification Head for narrative similarity.

Combines frozen SBERT embeddings with learned similarity features
to classify which text is more similar to the anchor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import Optional, Tuple


class SiameseClassifier(nn.Module):
    """
    Siamese Network with Classification Head.
    
    Architecture:
    1. SBERT Encoder: Generates embeddings for anchor, text_a, text_b
    2. Feature Extraction: Computes difference vectors and cosine similarities
    3. Classification Head: Linear layers to predict which text is more similar
    
    Input: (anchor_text, text_a, text_b) - strings
    Output: logits (2,) - raw predictions for [text_a closer, text_b closer]
    """
    
    def __init__(
        self,
        sbert_model_path: str = "all-MiniLM-L6-v2",
        hidden_dim: int = 256,
        dropout_rate: float = 0.2,
        freeze_sbert: bool = True,
        embedding_dim: int = 384
    ):
        """
        Initialize Siamese Classifier.
        
        Args:
            sbert_model_path: Path to fine-tuned SBERT model
            hidden_dim: Hidden dimension for classification head
            dropout_rate: Dropout rate for regularization
            freeze_sbert: Whether to freeze SBERT parameters
            embedding_dim: Dimension of SBERT embeddings (384 for all-MiniLM-L6-v2)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Load SBERT encoder
        print(f"Loading SBERT model from {sbert_model_path}...")
        self.sbert = SentenceTransformer(sbert_model_path)
        
        # Freeze SBERT if specified
        if freeze_sbert:
            for param in self.sbert.parameters():
                param.requires_grad = False
            print("SBERT parameters frozen")
        else:
            print("SBERT parameters trainable")
        
        # Feature dimension: [diff_a (384) + diff_b (384) + sim_a (1) + sim_b (1)]
        self.feature_dim = embedding_dim * 2 + 2
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2)
        )
        
        print(f"Feature dimension: {self.feature_dim}")
        print(f"Hidden dimension: {hidden_dim}")
    
    def encode_texts(self, texts: list) -> torch.Tensor:
        """
        Encode list of texts using SBERT.
        
        Args:
            texts: List of strings to encode
            
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        embeddings = self.sbert.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        return embeddings
    
    def compute_features(
        self,
        anchor_emb: torch.Tensor,
        text_a_emb: torch.Tensor,
        text_b_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract similarity features from embeddings.
        
        Features: [diff_a, diff_b, sim_a, sim_b]
        
        Args:
            anchor_emb: Anchor embeddings (batch_size, embedding_dim)
            text_a_emb: Text A embeddings (batch_size, embedding_dim)
            text_b_emb: Text B embeddings (batch_size, embedding_dim)
            
        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        # Compute difference vectors
        diff_a = anchor_emb - text_a_emb  # (batch_size, embedding_dim)
        diff_b = anchor_emb - text_b_emb  # (batch_size, embedding_dim)
        
        # Compute cosine similarities
        sim_a = F.cosine_similarity(
            anchor_emb, text_a_emb, dim=-1, eps=1e-8
        ).unsqueeze(-1)  # (batch_size, 1)
        
        sim_b = F.cosine_similarity(
            anchor_emb, text_b_emb, dim=-1, eps=1e-8
        ).unsqueeze(-1)  # (batch_size, 1)
        
        # Concatenate all features
        features = torch.cat([diff_a, diff_b, sim_a, sim_b], dim=-1)
        
        return features
    
    def forward(
        self,
        anchor_texts: list,
        text_a_list: list,
        text_b_list: list
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            anchor_texts: List of anchor texts
            text_a_list: List of text A samples
            text_b_list: List of text B samples
            
        Returns:
            Logits tensor of shape (batch_size, 2)
            logits[:, 0] = score for "text_a closer"
            logits[:, 1] = score for "text_b closer"
        """
        # Encode all texts
        with torch.no_grad():
            anchor_emb = self.encode_texts(anchor_texts)
            text_a_emb = self.encode_texts(text_a_list)
            text_b_emb = self.encode_texts(text_b_list)
        
        # Extract features
        features = self.compute_features(anchor_emb, text_a_emb, text_b_emb)
        
        # Pass through classifier
        logits = self.classifier(features)
        
        return logits
    
    def predict(
        self,
        anchor_texts: list,
        text_a_list: list,
        text_b_list: list,
        return_probs: bool = False
    ) -> torch.Tensor:
        """
        Make predictions in eval mode.
        
        Args:
            anchor_texts: List of anchor texts
            text_a_list: List of text A samples
            text_b_list: List of text B samples
            return_probs: If True, return probabilities instead of predictions
            
        Returns:
            If return_probs=True: Probabilities (batch_size, 2)
            If return_probs=False: Predictions (batch_size,) - 0 or 1
        """
        self.eval()
        
        with torch.no_grad():
            logits = self.forward(anchor_texts, text_a_list, text_b_list)
            
            if return_probs:
                probs = F.softmax(logits, dim=-1)
                return probs
            else:
                predictions = torch.argmax(logits, dim=-1)
                return predictions
    
    def get_trainable_params(self):
        """
        Get trainable parameters (only classifier if SBERT is frozen).
        
        Returns:
            List of trainable parameters
        """
        return [p for p in self.parameters() if p.requires_grad]
    
    def count_parameters(self) -> dict:
        """
        Count model parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": frozen_params
        }


if __name__ == "__main__":
    # Quick test
    print("Testing SiameseClassifier...")
    
    # Initialize model
    model = SiameseClassifier(
        sbert_model_path="output/sbert_finetuned_modelepoch10",
        hidden_dim=256,
        dropout_rate=0.2,
        freeze_sbert=True
    )
    
    # Print parameter count
    param_counts = model.count_parameters()
    print(f"\nParameter counts:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Trainable: {param_counts['trainable']:,}")
    print(f"  Frozen: {param_counts['frozen']:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    anchor = ["A man walks into a bar"]
    text_a = ["A person enters a pub"]
    text_b = ["A dog runs in the forest"]
    
    model.eval()
    with torch.no_grad():
        logits = model.forward(anchor, text_a, text_b)
        print(f"Logits shape: {logits.shape}")
        print(f"Logits: {logits}")
        
        probs = model.predict(anchor, text_a, text_b, return_probs=True)
        print(f"Probabilities: {probs}")
        
        pred = model.predict(anchor, text_a, text_b, return_probs=False)
        print(f"Prediction: {pred.item()} (0=text_a closer, 1=text_b closer)")
