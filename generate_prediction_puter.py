import pandas as pd
from puter_api.similarity_prompt import choose_more_similar_prompt

df = pd.read_csv("data.csv")

predictions = []

for i, row in df.iterrows():
    anchor = row["anchor_text"]
    a = row["option_a"]
    b = row["option_b"]

    pred = choose_more_similar_prompt(anchor, a, b)
    predictions.append(pred)

df["prediction"] = predictions
df.to_csv("result.csv", index=False)

print("Done.")
