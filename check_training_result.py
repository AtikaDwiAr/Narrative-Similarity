import torch

# Load checkpoint
checkpoint = torch.load('output/siamese_classifier_best.pth', map_location='cpu', weights_only=False)

print("="*60)
print("Training Results Summary")
print("="*60)
print(f"Best Epoch: {checkpoint.get('epoch', 'N/A') + 1}")
print(f"Best Val Accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
print(f"Total Epochs Trained: {len(checkpoint.get('train_losses', []))}")
print()
print("Training Progress:")
print("-"*60)
print(f"{'Epoch':<10} {'Train Loss':<15} {'Val Loss':<15} {'Val Acc':<15}")
print("-"*60)

train_losses = checkpoint.get('train_losses', [])
val_losses = checkpoint.get('val_losses', [])
val_accuracies = checkpoint.get('val_accuracies', [])

for i in range(len(train_losses)):
    marker = " ← BEST" if i == checkpoint.get('epoch', -1) else ""
    print(f"{i+1:<10} {train_losses[i]:<15.4f} {val_losses[i]:<15.4f} {val_accuracies[i]:<15.4f}{marker}")

print("="*60)
