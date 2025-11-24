"""
Prediction script for Siamese Classifier on test set.
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm

from model_siamese_classifier import SiameseClassifier
from dataset_classifier import NarrativeSimilarityDataset
from utils import load_jsonl, save_jsonl, get_device, set_seed


def load_model(model_path: str, device: str) -> SiameseClassifier:
    """
    Load trained Siamese model.
    
    Args:
        model_path: Path to saved model checkpoint
        device: Device to load model to
        
    Returns:
        Loaded model in eval mode
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    
    # Initialize model architecture
    model = SiameseClassifier(
        sbert_model_path="all-MiniLM-L6-v2",
        hidden_dim=256,
        dropout_rate=0.2,
        freeze_sbert=True
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Best epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best val accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
    
    return model


def predict_on_batch(
    model: SiameseClassifier,
    anchors: list,
    texts_a: list,
    texts_b: list
) -> torch.Tensor:
    """
    Make predictions on a batch of samples.
    
    Args:
        model: Siamese classifier model
        anchors: List of anchor texts
        texts_a: List of text A samples
        texts_b: List of text B samples
        
    Returns:
        Predictions (0 or 1)
    """
    with torch.no_grad():
        predictions = model.predict(
            anchor_texts=anchors,
            text_a_list=texts_a,
            text_b_list=texts_b,
            return_probs=False
        )
    
    return predictions


def predict_on_test_set(
    model_path: str,
    test_data_path: str,
    output_path: str = "output/track_a_siamese.jsonl",
    batch_size: int = 32
) -> None:
    """
    Make predictions on entire test set and save results.
    
    Args:
        model_path: Path to saved model
        test_data_path: Path to test data JSONL
        output_path: Path to save predictions
        batch_size: Batch size for inference
    """
    # Setup
    device = get_device()
    set_seed(42)
    
    # Load model
    model = load_model(model_path, device)
    
    # Load test data
    print(f"\nLoading test data from {test_data_path}...")
    test_data = load_jsonl(test_data_path)
    print(f"✓ Loaded {len(test_data)} test samples")
    
    # Make predictions
    print(f"\nMaking predictions...")
    predictions = []
    
    for batch_idx in tqdm(range(0, len(test_data), batch_size), desc="Predicting"):
        batch_end = min(batch_idx + batch_size, len(test_data))
        batch_data = test_data[batch_idx:batch_end]
        
        # Extract texts
        anchors = [item["anchor_text"] for item in batch_data]
        texts_a = [item["text_a"] for item in batch_data]
        texts_b = [item["text_b"] for item in batch_data]
        
        # Get predictions
        batch_predictions = predict_on_batch(model, anchors, texts_a, texts_b)
        
        # Convert predictions to results
        for i, pred in enumerate(batch_predictions):
            pred_int = pred.item()
            # Label 0 = text_a closer → True
            # Label 1 = text_b closer → False
            text_a_is_closer = (pred_int == 0)
            
            result = {
                "anchor_text": batch_data[i]["anchor_text"],
                "text_a": batch_data[i]["text_a"],
                "text_b": batch_data[i]["text_b"],
                "text_a_is_closer": text_a_is_closer
            }
            predictions.append(result)
    
    # Save predictions
    save_jsonl(predictions, output_path)
    
    print(f"\n✓ Predictions saved to {output_path}")
    print(f"Total predictions: {len(predictions)}")
    
    # Print statistics
    count_a_closer = sum(1 for p in predictions if p["text_a_is_closer"])
    count_b_closer = len(predictions) - count_a_closer
    
    print(f"\nPrediction statistics:")
    print(f"  Text A closer: {count_a_closer} ({100*count_a_closer/len(predictions):.2f}%)")
    print(f"  Text B closer: {count_b_closer} ({100*count_b_closer/len(predictions):.2f}%)")


def evaluate_predictions(
    predictions_path: str,
    ground_truth_path: str
) -> dict:
    """
    Evaluate predictions against ground truth.
    
    Args:
        predictions_path: Path to predictions JSONL
        ground_truth_path: Path to ground truth JSONL
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating predictions...")
    
    # Load data
    predictions = load_jsonl(predictions_path)
    ground_truth = load_jsonl(ground_truth_path)
    
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Mismatch in data length: "
            f"predictions={len(predictions)}, ground_truth={len(ground_truth)}"
        )
    
    # Compute accuracy
    correct = sum(
        1 for p, g in zip(predictions, ground_truth)
        if p["text_a_is_closer"] == g["text_a_is_closer"]
    )
    
    accuracy = correct / len(predictions)
    
    results = {
        "total": len(predictions),
        "correct": correct,
        "incorrect": len(predictions) - correct,
        "accuracy": accuracy
    }
    
    print(f"✓ Evaluation complete")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{len(predictions)})")
    
    return results


def main():
    """
    Main prediction function.
    """
    # Configuration
    MODEL_PATH = "output/siamese_classifier_best.pth"
    TEST_DATA_PATH = "data/test_track_a.jsonl"
    OUTPUT_PATH = "output/track_a_siamese.jsonl"
    BATCH_SIZE = 32
    
    print(f"\n{'='*60}")
    print(f"Siamese Classifier - Test Set Prediction")
    print(f"{'='*60}")
    print(f"Model: {MODEL_PATH}")
    print(f"Test data: {TEST_DATA_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"{'='*60}\n")
    
    # Make predictions
    predict_on_test_set(
        model_path=MODEL_PATH,
        test_data_path=TEST_DATA_PATH,
        output_path=OUTPUT_PATH,
        batch_size=BATCH_SIZE
    )
    
    # Evaluate
    try:
        eval_results = evaluate_predictions(
            predictions_path=OUTPUT_PATH,
            ground_truth_path=TEST_DATA_PATH
        )
        
        print(f"\n{'='*60}")
        print(f"Evaluation Results:")
        print(f"{'='*60}")
        print(f"Accuracy: {eval_results['accuracy']:.4f}")
        print(f"Correct: {eval_results['correct']}/{eval_results['total']}")
        print(f"Incorrect: {eval_results['incorrect']}/{eval_results['total']}")
        print(f"{'='*60}\n")
    
    except Exception as e:
        print(f"\nWarning: Could not evaluate predictions: {e}")
        print("This is normal if ground truth labels are not in test data.")


if __name__ == "__main__":
    main()
