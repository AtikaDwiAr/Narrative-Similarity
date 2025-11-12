from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from torch.utils.data import DataLoader
import json
from tqdm import tqdm

# Load data
train_file = "data/train_track_a.jsonl"
data = [json.loads(line) for line in open(train_file, "r", encoding="utf-8")]


# Prepare data training dalam format InputExample
train_samples = []
skipped = 0

for row in tqdm(data, desc="Prepare data training"):
    anchor, text_a, text_b, label = (
        row["anchor_text"],
        row["text_a"],
        row["text_b"],
        row["text_a_is_closer"],
    )

    # Skip baris kosong / None
    if not all([anchor, text_a, text_b]):
        skipped += 1
        continue

    # label True berarti text_a lebih mirip → 1, jika tidak → 0
    label_a = float(1 if label else 0)
    label_b = float(1 if not label else 0)


    train_samples.append(InputExample(texts=[anchor, text_a], label=label_a))
    train_samples.append(InputExample(texts=[anchor, text_b], label=label_b))

# Inisialisasi model Sentence-BERT dasar
model = SentenceTransformer("all-MiniLM-L6-v2")

# DataLoader dan loss function
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# Training
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,                     
    warmup_steps=100,
    show_progress_bar=True
)

# Simpan model
model.save("output/sbert_finetuned_modelepoch10")
print("Model stored in output/sbert_finetuned_modelepoch10")
