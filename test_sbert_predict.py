from sentence_transformers import SentenceTransformer, util
import json
from tqdm import tqdm
import os

# Load model
model_path = "output/sbert_finetuned_model"
model = SentenceTransformer(model_path)

# Load data test
test_file = "data/test_track_a.jsonl"
data = [json.loads(line) for line in open(test_file, "r", encoding="utf-8")]

# Pastikan folder output ada
os.makedirs("output", exist_ok=True)
output_path = "output/track_a.jsonl"

# Prediksi
with open(output_path, "w", encoding="utf-8") as fout:
    for row in tqdm(data, desc="Predicting"):
        anchor, text_a, text_b = row["anchor_text"], row["text_a"], row["text_b"]

        # Hitung cosine similarity
        emb_anchor = model.encode(anchor, convert_to_tensor=True)
        emb_a = model.encode(text_a, convert_to_tensor=True)
        emb_b = model.encode(text_b, convert_to_tensor=True)

        sim_a = util.cos_sim(emb_anchor, emb_a).item()
        sim_b = util.cos_sim(emb_anchor, emb_b).item()

        is_a_closer = sim_a > sim_b

        result = {
            "anchor_text": anchor,
            "text_a": text_a,
            "text_b": text_b,
            "text_a_is_closer": is_a_closer
        }
        fout.write(json.dumps(result) + "\n")

print(f"Hasil disimpan di: {output_path}")
