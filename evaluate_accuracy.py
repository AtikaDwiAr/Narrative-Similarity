import json
from pathlib import Path
from typing import Dict, List, Tuple


def compute_accuracy(predictions: List[dict], ground_truth: List[dict]) -> Tuple[int, float]:
    """
    Compute accuracy of predictions against ground truth.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        
    Returns:
        Tuple of (correct_count, accuracy)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Data length mismatch: predictions={len(predictions)}, "
            f"ground_truth={len(ground_truth)}"
        )
    
    correct = 0
    for p, g in zip(predictions, ground_truth):
        if p["text_a_is_closer"] == g["text_a_is_closer"]:
            correct += 1
    
    accuracy = correct / len(ground_truth)
    return correct, accuracy


def evaluate_single_model(
    pred_file: str,
    gold_file: str,
    model_name: str = "Model"
) -> Dict[str, any]:
    """
    Evaluate a single model's predictions.
    
    Args:
        pred_file: Path to prediction JSONL file
        gold_file: Path to ground truth JSONL file
        model_name: Name of the model
        
    Returns:
        Dictionary with evaluation results
    """
    # Load data
    with open(pred_file, "r", encoding="utf-8") as f:
        preds = [json.loads(line) for line in f]
    
    with open(gold_file, "r", encoding="utf-8") as f:
        golds = [json.loads(line) for line in f]
    
    # Compute accuracy
    correct, accuracy = compute_accuracy(preds, golds)
    
    return {
        "model_name": model_name,
        "pred_file": pred_file,
        "gold_file": gold_file,
        "total": len(golds),
        "correct": correct,
        "incorrect": len(golds) - correct,
        "accuracy": accuracy
    }


def compare_models(
    models: List[Tuple[str, str]],
    gold_file: str
) -> None:
    """
    Compare multiple models' predictions.
    
    Args:
        models: List of tuples (pred_file, model_name)
        gold_file: Path to ground truth JSONL file
    """
    results = []
    
    print(f"\n{'='*80}")
    print(f"{'Model Comparison':^80}")
    print(f"{'='*80}")
    print(f"Ground truth: {gold_file}\n")
    
    for pred_file, model_name in models:
        if not Path(pred_file).exists():
            print(f"⚠ Warning: File not found: {pred_file}")
            continue
        
        result = evaluate_single_model(pred_file, gold_file, model_name)
        results.append(result)
        
        print(f"{model_name}:")
        print(f"  File: {pred_file}")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Correct: {result['correct']}/{result['total']}\n")
    
    # Print comparison table
    if len(results) > 1:
        print(f"\n{'='*80}")
        print(f"{'Model':<30} {'Accuracy':<20} {'Correct':<20}")
        print(f"{'-'*80}")
        
        for result in results:
            accuracy_pct = result['accuracy'] * 100
            correct_str = f"{result['correct']}/{result['total']}"
            print(f"{result['model_name']:<30} {accuracy_pct:>18.4f}% {correct_str:>20}")
        
        print(f"{'-'*80}")
        
        # Find best model
        best_result = max(results, key=lambda x: x['accuracy'])
        print(f"\n✓ Best model: {best_result['model_name']} ({best_result['accuracy']:.4f})")
        
        # Calculate improvements
        if len(results) >= 2:
            sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
            for i in range(1, len(sorted_results)):
                improvement = sorted_results[i-1]['accuracy'] - sorted_results[i]['accuracy']
                print(f"  {sorted_results[i-1]['model_name']} vs {sorted_results[i]['model_name']}: "
                      f"+{improvement:.4f} ({improvement*100:.2f}%)")
    
    print(f"{'='*80}\n")


# Main execution
if __name__ == "__main__":
    # Single model evaluation (default behavior)
    pred_file = "output/track_a.jsonl"
    gold_file = "data/test_track_a.jsonl"
    
    if Path(pred_file).exists():
        print(f"\n{'='*80}")
        print(f"Evaluating: {pred_file}")
        print(f"{'='*80}\n")
        
        result = evaluate_single_model(pred_file, gold_file, "Baseline (CosineSimilarity)")
        
        print(f"Jumlah benar: {result['correct']}/{result['total']}")
        print(f"Akurasi: {result['accuracy']:.4f}")
        print(f"{'='*80}\n")
    else:
        print(f"Warning: Baseline prediction file not found: {pred_file}")
    
    # Compare multiple models if available
    siamese_pred = "output/track_a_siamese.jsonl"
    
    if Path(siamese_pred).exists():
        print("\nComparing multiple models...\n")
        
        models_to_compare = [
            (pred_file, "Baseline (CosineSimilarity)") if Path(pred_file).exists() else None,
            (siamese_pred, "Siamese Classifier")
        ]
        
        models_to_compare = [m for m in models_to_compare if m is not None]
        
        if len(models_to_compare) > 1:
            compare_models(models_to_compare, gold_file)

