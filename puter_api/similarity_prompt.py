from .client import PuterClient

client = PuterClient()

def choose_more_similar_prompt(anchor: str, text_a: str, text_b: str) -> str:
    prompt = f"""
Anchor: {anchor}

A: {text_a}

B: {text_b}

Which text is more similar to the anchor? Answer with only 'A' or 'B'.
"""
    res = client.chat(prompt).strip()
    return res
