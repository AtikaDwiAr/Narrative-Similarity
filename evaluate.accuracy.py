import json

# Path file
pred_file = "output/track_a.jsonl"   # hasil prediksi
gold_file = "data/test_track_a.jsonl" # file ground truth

# Load data
with open(pred_file, "r", encoding="utf-8") as f:
    preds = [json.loads(line) for line in f]

with open(gold_file, "r", encoding="utf-8") as f:
    golds = [json.loads(line) for line in f]

# Validasi panjang data
assert len(preds) == len(golds), f"Panjang tidak sama: pred={len(preds)}, gold={len(golds)}"

# Hitung akurasi
correct = 0
for i, (p, g) in enumerate(zip(preds, golds)):
    # Compare hasil prediksi vs label ground truth
    if p["text_a_is_closer"] == g["text_a_is_closer"]:
        correct += 1

accuracy = correct / len(golds)

# Hasil
print(f"Jumlah benar: {correct}/{len(golds)}")
print(f"Akurasi: {accuracy:.4f}")
