import requests

class PuterClient:
    def __init__(self, host="http://localhost:5353"):
        self.host = host

    def chat(self, prompt: str) -> str:
        url = f"{self.host}/v1/chat/generate"

        payload = {
            "model": "llama-3.2-3b-instruct",
            "messages": [{"role": "user", "content": prompt}],
        }

        r = requests.post(url, json=payload)
        r.raise_for_status()

        return r.json()["choices"][0]["message"]["content"]
